/**
 * \file src/serialization/include/megbrain/serialization/opr_load_dump.h
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 *
 */

#pragma once

#include "megbrain/graph.h"
#include "megbrain/serialization/load_dump_config.h"
#include "megbrain/serialization/opr_registry.h"
#if MGB_ENABLE_FBS_SERIALIZATION
#include <flatbuffers/flatbuffers.h>
#endif

namespace mgb {
namespace serialization {

namespace fbs {
template <typename T>
struct OperatorParamTraits;
}

enum class SerializationFormat {
    RAW_POD,
#if MGB_ENABLE_FBS_SERIALIZATION
    FLATBUFFERS,
#endif
};

//! context for serializing a single operator
class OprDumpContext {
    const SerializationFormat m_format;

    OprDumpContext(const SerializationFormat fmt) : m_format{fmt} {}
    friend class OprDumpContextRawPOD;
    friend class OprDumpContextFlatBuffers;

protected:
    virtual ~OprDumpContext() = default;

public:
    enum class TensorWriteMethod {
        META_INPUT,       //!< only meta info, for graph input with name
        VALUE_INPUT,      //!< value for H2D as graph inp with name
        VALUE_SHARED,     //!< shared tensor; load by load_tensor_shared()
        VALUE_ANONYMOUS,  //!< value without name
    };

    /*!
     * \brief write value (or only meta info) of a tensor to the output
     *      stream
     * \param name name used for retrieving the tensor after loading;
     *      pass an empty string to disable retrieving by name
     */
    virtual void dump_tensor(const std::string& name,
                             const HostTensorND& tensor,
                             TensorWriteMethod method) = 0;

    //! get associated global configuration
    virtual const GraphDumpConfig& config() const = 0;

    //! write a buffer with its length
    virtual void dump_buf_with_len(const void* data, uint32_t size) = 0;

    /*!
     * \brief write a param tag with its value
     */
    template <class Param>
    void write_param(const Param& param);
};

class OprDumpContextRawPOD : public OprDumpContext {
    const bool m_check_param_tag;

protected:
    OprDumpContextRawPOD(bool check_param_tag = true)
            : OprDumpContext(SerializationFormat::RAW_POD),
              m_check_param_tag{check_param_tag} {}
    //! write to the output stream
    virtual void write_raw(const void* data, size_t size) = 0;

public:
    void dump_buf_with_len(const void* data, uint32_t size) override final {
        write_raw(&size, sizeof(size));
        write_raw(data, size);
    }

    /*!
     * \brief write a param tag with its value
     */
    template <class Param>
    void write_param(const Param& param) {
        static_assert(is_location_invariant<Param>::value,
                      "param must be location-invariant");
        if (m_check_param_tag) {
            uint32_t tag = Param::TAG;
            write_raw(&tag, sizeof(tag));
        }
        write_raw(&param, sizeof(Param));
    }
};

template <>
void OprDumpContextRawPOD::write_param(const DType& param);

namespace fbs {
template <typename T>
struct ParamConverter {};

struct Yes {};
struct No {};
template <typename T>
struct SupportFlatBuffersSerialization : Yes {};
}  // namespace fbs

#if MGB_ENABLE_FBS_SERIALIZATION
class OprDumpContextFlatBuffers : public OprDumpContext {
protected:
    OprDumpContextFlatBuffers()
            : OprDumpContext(SerializationFormat::FLATBUFFERS) {}
    virtual void append_param(uint32_t type,
                              flatbuffers::Offset<void> value) = 0;

public:
    virtual flatbuffers::FlatBufferBuilder& builder() = 0;

    template <class Param>
    void write_param(const Param& param, fbs::Yes) {
        using ResultType = typename fbs::ParamConverter<Param>::FlatBufferType;
        static_assert(fbs::OperatorParamTraits<ResultType>::enum_value != 0,
                      "invalid param");
        auto param_offset =
                fbs::ParamConverter<Param>::to_flatbuffer(builder(), param);
        append_param(fbs::OperatorParamTraits<ResultType>::enum_value,
                     param_offset.Union());
    }

    template <class Param>
    void write_param(const Param& param, fbs::No) {
        mgb_throw(SerializationError,
                  "Serialization of operator param %s unsupported", __func__);
    }
};
#endif

template <class Param>
void OprDumpContext::write_param(const Param& p) {
    static_assert(is_location_invariant<Param>::value,
                  "param must be location-invariant");
    switch (m_format) {
        case SerializationFormat::RAW_POD:
            static_cast<OprDumpContextRawPOD*>(this)->write_param(p);
            break;
#if MGB_ENABLE_FBS_SERIALIZATION
        case SerializationFormat::FLATBUFFERS:
            static_cast<OprDumpContextFlatBuffers*>(this)->write_param(
                    p, fbs::SupportFlatBuffersSerialization<Param>{});
            break;
#endif
    }
}

/*!
 * \brief context for deserializing a single operator
 *
 * Note that this class is also a UserData, and it can be accessed from the
 * graph by querying its ComputingGraph::options().user_data
 */
class OprLoadContext : public UserDataContainer::UserData {
    MGB_TYPEINFO_OBJ_DECL;
    const SerializationFormat m_format;
    const uint32_t m_mgb_version;

    explicit OprLoadContext(const SerializationFormat fmt,
                            uint32_t mgb_version = 0)
            : m_format{fmt}, m_mgb_version{mgb_version} {}

    friend class OprLoadContextRawPOD;
    friend class OprLoadContextFlatBuffers;

public:
    //! get current computing graph
    virtual ComputingGraph& graph() = 0;

    //! load tensor; a new instance is created each time
    virtual std::shared_ptr<HostTensorND> load_tensor() = 0;

    /*!
     * \brief load shared tensor
     *
     * previous instance would be reused if possible.
     *
     * It must be dumped with TensorWriteMethod::VALUE_SHARED
     */
    virtual std::shared_ptr<DeviceTensorND> load_tensor_shared() = 0;

    //! get associated global configuration
    virtual const GraphLoadConfig& config() const = 0;

    //! make an opr loader from given identifier
    OprLoader make_opr_loader(const std::string& id);

    /*!
     * \brief load a buffer dumped by OprDumpContext::dump_buf_with_len
     *
     * Alignment same as data buffer in std::string.
     */
    virtual std::string load_buf_with_len() = 0;

    /*!
     * \brief like load_buf_with_len(), but share the buffer with
     *      underlying storage if possible
     *
     * Note that there is not alignment gurantee.
     */
    virtual SharedBuffer load_shared_buf_with_len() = 0;

    /*!
     * \brief read a param and check that tag matches
     */
    template <class Param>
    Param read_param();

    /*!
     * \brief the version of MegBrain that produced the model dump
     *
     * The value is 0 if version info is unavailable due to either
     * reading a legacy model file or loading from python generated
     * temporary models during graph construction.
     */
    uint32_t mgb_version() const { return m_mgb_version; }
};

class OprLoadContextRawPOD : public OprLoadContext {
    const bool m_check_param_tag;
    template <typename T>
    struct ParamPack {
        uint32_t tag;
        uint8_t param[sizeof(T)];
    } MGB_PACKED;

protected:
    explicit OprLoadContextRawPOD(bool check_param_tag = true,
                                  uint32_t mgb_version = 0)
            : OprLoadContext(SerializationFormat::RAW_POD, mgb_version),
              m_check_param_tag{check_param_tag} {}

    virtual void read_raw(void* dest, size_t size) = 0;

    //! used for implementing load_shared_buf_with_len(); the default
    //! implementation uses read_raw()
    virtual SharedBuffer load_shared_buf(size_t size);

public:
    std::string load_buf_with_len() override;

    SharedBuffer load_shared_buf_with_len() override;

    template <class Param>
    Param read_param() {
        static_assert(is_location_invariant<Param>::value,
                      "param must be location-invariant");
        std::aligned_storage_t<sizeof(Param), alignof(Param)> p;
        if (m_check_param_tag) {
            ParamPack<Param> pack;
            read_raw(&pack, sizeof(pack));
            mgb_assert(pack.tag == Param::TAG);
            memcpy(&p, pack.param, sizeof(Param));
        } else {
            read_raw(&p, sizeof(p));
        }
        return *aliased_ptr<Param>(&p);
    }
};

template <>
DType OprLoadContextRawPOD::read_param();

#if MGB_ENABLE_FBS_SERIALIZATION
class OprLoadContextFlatBuffers : public OprLoadContext {
protected:
    explicit OprLoadContextFlatBuffers(uint32_t mgb_version = 0)
            : OprLoadContext(SerializationFormat::FLATBUFFERS, mgb_version) {}
    virtual const void* get_next_param(uint32_t enumv) = 0;

public:
    std::string load_buf_with_len() override = 0;
    SharedBuffer load_shared_buf_with_len() override = 0;

    template <class T>
    T read_param(fbs::Yes) {
        using SourceType = typename fbs::ParamConverter<T>::FlatBufferType;
        auto p = get_next_param(
                fbs::OperatorParamTraits<SourceType>::enum_value);
        mgb_assert(p != nullptr, "wrong param type");
        return fbs::ParamConverter<T>::to_param(
                static_cast<const SourceType*>(p));
    }

    template <class T>
    T read_param(fbs::No) {
        mgb_throw(SerializationError,
                  "Deserialization of operator param %s unsupported", __func__);
    }
};
#endif

template <class Param>
Param OprLoadContext::read_param() {
    switch (m_format) {
        case SerializationFormat::RAW_POD:
            return static_cast<OprLoadContextRawPOD*>(this)
                    ->read_param<Param>();
#if MGB_ENABLE_FBS_SERIALIZATION
        case SerializationFormat::FLATBUFFERS:
            return static_cast<OprLoadContextFlatBuffers*>(this)
                    ->read_param<Param>(
                            fbs::SupportFlatBuffersSerialization<Param>{});
#endif
    }
    mgb_assert(0);
}

}  // namespace serialization
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
