#pragma once

#include "megbrain/serialization/opr_load_dump.h"
#include "megbrain/serialization/opr_registry.h"
#include "megbrain/serialization/opr_shallow_copy.h"
#include "megbrain/utils/hash_ct.h"

namespace mgb {
namespace serialization {

/*!
 * \brief get persistent param used for InputStream and OutputStream, and
 *      can be conveted from/to Opr::Param
 *
 * used by OprLoadDumpImpl
 */
template <class Opr>
struct OprPersistentParam {
    using Param = typename Opr::Param;
};

/*!
 * \brief used by opr_loader_general to create opr instance; arity has been
 *      checked before calling its make() method
 */
template <class Opr, size_t arity>
struct OprMaker;

//! OprMaker implementation for operators with variadic arguments
template <class Opr>
struct OprMakerVariadic {
    using Param = typename Opr::Param;
    static cg::OperatorNodeBase* make(
            const Param& param, const cg::VarNodeArray& inputs, ComputingGraph& graph,
            const OperatorNodeConfig& config) {
        MGB_MARK_USED_VAR(graph);
        return Opr::make(inputs, param, config).node()->owner_opr();
    }
};

/*!
 * \tparam arity number of input vars; pass 0 for a custom impl
 */
template <class Opr, size_t arity>
struct OprLoadDumpImpl {
    using PersisParam = typename OprPersistentParam<Opr>::Param;

    //! a general operator dumper by writing its param as POD
    static void dump(OprDumpContext& ctx, const cg::OperatorNodeBase& opr) {
        ctx.write_param<PersisParam>(opr.cast_final_safe<Opr>().param());
    }

    /*!
     * \brief loader corresponding to dump()
     *
     * OprMaker<> would be used to create the opr
     */
    static cg::OperatorNodeBase* load(
            OprLoadContext& ctx, const cg::VarNodeArray& inputs,
            const OperatorNodeConfig& config) {
        if (arity) {
            mgb_assert(inputs.size() == arity);
        }
        return OprMaker<Opr, arity>::make(
                ctx.read_param<PersisParam>(), inputs, ctx.graph(), config);
    }
};

template <class Opr, size_t arity>
struct OprLoadDumpImplV2 : public OprLoadDumpImpl<Opr, arity> {};

#define IMPL_OPR_MAKER(_arity, _args...)                                              \
    template <class Opr>                                                              \
    struct OprMaker<Opr, _arity> {                                                    \
        using Param = typename Opr::Param;                                            \
        static cg::OperatorNodeBase* make(                                            \
                const Param& param, const cg::VarNodeArray& i, ComputingGraph& graph, \
                const OperatorNodeConfig& config) {                                   \
            MGB_MARK_USED_VAR(param);                                                 \
            MGB_MARK_USED_VAR(i);                                                     \
            MGB_MARK_USED_VAR(graph);                                                 \
            return Opr::make(_args, config).node()->owner_opr();                      \
        }                                                                             \
    };
IMPL_OPR_MAKER(1, i[0], param);
IMPL_OPR_MAKER(2, i[0], i[1], param);
IMPL_OPR_MAKER(3, i[0], i[1], i[2], param);
IMPL_OPR_MAKER(4, i[0], i[1], i[2], i[3], param);
IMPL_OPR_MAKER(5, i[0], i[1], i[2], i[3], i[4], param);
#undef IMPL_OPR_MAKER

/*!
 * \brief a template to call Callee::entry()
 *
 * This can be partially specialized to omit registry entries for some oprs
 */
template <class Callee>
struct OprRegistryCallerDefaultImpl {
    OprRegistryCallerDefaultImpl() { Callee::entry(); }
};

#ifdef MGB_OPR_REGISTRY_CALLER_SPECIALIZE
MGB_OPR_REGISTRY_CALLER_SPECIALIZE
#else
template <class Opr, class Callee>
struct OprRegistryCaller : public OprRegistryCallerDefaultImpl<Callee> {};
#endif

}  // namespace serialization
}  // namespace mgb

#if MGB_VERBOSE_TYPEINFO_NAME
//! name of operator from class
#define _MGB_SEREG_OPR_NAME_FROM_CLS(_cls) #_cls
#else
#define _MGB_SEREG_OPR_NAME_FROM_CLS(_cls) \
    {}
#endif

/*!
 * \brief call _impl::entry() on global initialization if OprRegistryCaller is
 *      not specilized for this opr
 */
#define MGB_SEREG_OPR_INTL_CALL_ENTRY(_cls, _impl)                       \
    namespace {                                                          \
    [[gnu::unused]] ::mgb::serialization::OprRegistryCaller<_cls, _impl> \
            __caller_OprReg##_cls##_ins;                                 \
    }

#define MGB_SEREG_OPR_INTL_CALL_ENTRY_V2(_cls, _impl)                    \
    namespace {                                                          \
    [[gnu::unused]] ::mgb::serialization::OprRegistryCaller<_cls, _impl> \
            __caller_V2_OprReg##_cls##_ins;                              \
    }

// Trim the terminating null character and a "V0" like suffix from the string
// then hash it.
// TODO: Get rid of this.
#define MGB_HASH_STR_WITHOUT_TAIL_0_AND_VERSION(v)               \
    ::mgb::EnsureHashConstexpr<::mgb::XXHash64CT::hash(          \
            v,                                                   \
            sizeof(v) - 1 -                                      \
                    (sizeof(v) > 2 && v[sizeof(v) - 2] >= '0' && \
                                     v[sizeof(v) - 2] <= '9' &&  \
                                     v[sizeof(v) - 3] == 'V'     \
                             ? 2                                 \
                             : 0),                               \
            20160701)>::val

//! call OprRegistry::add for old serialization
//! call OprRegistryV2::versioned_add for new serialization which is compatiable
//! with old serialization, convert is nullptr, this registry is just only for
//! varsion 1
#define MGB_SEREG_OPR_INTL_CALL_ADD(_cls, _dump, _load, _registerv2)                   \
    do {                                                                               \
        ::mgb::serialization::OprRegistry::add(                                        \
                {_cls::typeinfo(),                                                     \
                 MGB_HASH_STR(#_cls),                                                  \
                 _MGB_SEREG_OPR_NAME_FROM_CLS(_cls),                                   \
                 _dump,                                                                \
                 _load,                                                                \
                 {},                                                                   \
                 MGB_HASH_STR_WITHOUT_TAIL_0_AND_VERSION(#_cls)});                     \
        if (_registerv2) {                                                             \
            ::mgb::serialization::OprRegistryV2::versioned_add(                        \
                    {_cls::typeinfo(), MGB_HASH_STR_WITHOUT_TAIL_0_AND_VERSION(#_cls), \
                     _MGB_SEREG_OPR_NAME_FROM_CLS(_cls), _dump, _load, nullptr},       \
                    ::mgb::VERSION_1, ::mgb::VERSION_1);                               \
        }                                                                              \
    } while (0)

//! call OprRegistryV2::versioned_add for new serialization, in which convert the
//! function converter the Operator to the compatiable
#define MGB_SEREG_OPR_INTL_CALL_ADD_V2(                                       \
        _cls, _dump, _load, _convert, _version_min, _version_max)             \
    do {                                                                      \
        ::mgb::serialization::OprRegistryV2::versioned_add(                   \
                {_cls::typeinfo(), MGB_HASH_STR(#_cls),                       \
                 _MGB_SEREG_OPR_NAME_FROM_CLS(_cls), _dump, _load, _convert}, \
                _version_min, _version_max);                                  \
    } while (0)

/*!
 * \brief register opr serialization methods
 */
#define MGB_SEREG_OPR_CONDITION(_cls, _arity, _registerv2)                           \
    namespace {                                                                      \
    namespace ser = ::mgb::serialization;                                            \
    struct _OprReg##_cls {                                                           \
        using Impl = ser::OprLoadDumpImpl<_cls, _arity>;                             \
        static ser::OprWithOutputAccessor wrap_loader(                               \
                ser::OprLoadContext& ctx, const mgb::cg::VarNodeArray& inputs,       \
                const mgb::cg::OperatorNodeConfig& config) {                         \
            return ser::OprWithOutputAccessor(Impl::load(ctx, inputs, config));      \
        }                                                                            \
        static void entry() {                                                        \
            MGB_SEREG_OPR_INTL_CALL_ADD(_cls, Impl::dump, wrap_loader, _registerv2); \
        }                                                                            \
    };                                                                               \
    }                                                                                \
    MGB_SEREG_OPR_INTL_CALL_ENTRY(_cls, _OprReg##_cls)

#define MGB_SEREG_OPR(_cls, _arity) MGB_SEREG_OPR_CONDITION(_cls, _arity, true)

//! new dump/load function should implement in OprLoadDumpImplV2, _converter is
//! optional , if not implement pass nullptr
#define MGB_SEREG_OPR_V2(_cls, _arity, _converter, _version_min, _version_max)  \
    namespace {                                                                 \
    namespace ser = ::mgb::serialization;                                       \
    struct _OprRegV2##_cls {                                                    \
        using Impl = ser::OprLoadDumpImplV2<_cls, _arity>;                      \
        static ser::OprWithOutputAccessor wrap_loader(                          \
                ser::OprLoadContext& ctx, const mgb::cg::VarNodeArray& inputs,  \
                const mgb::cg::OperatorNodeConfig& config) {                    \
            return ser::OprWithOutputAccessor(Impl::load(ctx, inputs, config)); \
        }                                                                       \
        static void entry() {                                                   \
            MGB_SEREG_OPR_INTL_CALL_ADD_V2(                                     \
                    _cls, Impl::dump, wrap_loader, _converter, _version_min,    \
                    _version_max);                                              \
        }                                                                       \
    };                                                                          \
    }                                                                           \
    MGB_SEREG_OPR_INTL_CALL_ENTRY_V2(_cls, _OprRegV2##_cls)

//! use to check type is complete or not, midout need a complete type
template <class T, class = void>
struct IsComplete : std::false_type {};

template <class T>
struct IsComplete<T, decltype(void(sizeof(T)))> : std::true_type {};

//! call OprRegistry::add with only loader, used for backward compatibility
#define MGB_SEREG_OPR_COMPAT_WITH_ACCESSOR(_name, _load, _accessor)                    \
    namespace {                                                                        \
    static_assert(                                                                     \
            IsComplete<_name>(), "need a complete type for MGB_SEREG_OPR_COMPAT");     \
    namespace ser = ::mgb::serialization;                                              \
    struct _OprReg##_name {                                                            \
        static ser::OprWithOutputAccessor compat_loader(                               \
                ser::OprLoadContext& ctx, const mgb::cg::VarNodeArray& inputs,         \
                const mgb::cg::OperatorNodeConfig& config) {                           \
            auto&& ctx_ = static_cast<ser::OprLoadContext&>(ctx);                      \
            return ser::OprWithOutputAccessor(_load(ctx_, inputs, config), _accessor); \
        }                                                                              \
        static void entry() {                                                          \
            ser::OprRegistry::add(                                                     \
                    {nullptr,                                                          \
                     MGB_HASH_STR(#_name),                                             \
                     _MGB_SEREG_OPR_NAME_FROM_CLS(_name),                              \
                     nullptr,                                                          \
                     compat_loader,                                                    \
                     {},                                                               \
                     {}});                                                             \
        }                                                                              \
    };                                                                                 \
    }                                                                                  \
    MGB_SEREG_OPR_INTL_CALL_ENTRY(_name, _OprReg##_name)

#define MGB_SEREG_OPR_COMPAT(_name, _load) \
    MGB_SEREG_OPR_COMPAT_WITH_ACCESSOR(_name, _load, nullptr)

/*!
 * \brief use \p _copy to implement shallow copy for given operator
 */
#define MGB_REG_OPR_SHALLOW_COPY_IMPL(_cls, _copy)                                     \
    do {                                                                               \
        auto reg = ::mgb::serialization::OprRegistry::find_by_type(_cls::typeinfo());  \
        if (!reg) {                                                                    \
            ::mgb::serialization::OprRegistry::add(                                    \
                    {_cls::typeinfo(),                                                 \
                     MGB_HASH_STR(#_cls),                                              \
                     _MGB_SEREG_OPR_NAME_FROM_CLS(_cls),                               \
                     {},                                                               \
                     {},                                                               \
                     _copy,                                                            \
                     {}});                                                             \
        } else {                                                                       \
            const_cast<::mgb::serialization::OprRegistry*>(reg)->shallow_copy = _copy; \
        }                                                                              \
    } while (0)

/*!
 * \brief call MGB_REG_OPR_SHALLOW_COPY_IMPL on global initialization; if
 *      MGB_SEREG_OPR is also needed, this must be called after MGB_SEREG_OPR
 */
#define MGB_REG_OPR_SHALLOW_COPY(_cls, _copy)                               \
    namespace {                                                             \
    struct _OprRegShallowCopy##_cls {                                       \
        static void entry() { MGB_REG_OPR_SHALLOW_COPY_IMPL(_cls, _copy); } \
    };                                                                      \
    [[gnu::unused]] ::mgb::serialization::OprRegistryCaller<                \
            _cls, _OprRegShallowCopy##_cls>                                 \
            __caller_OprRegShallowCopy##_cls##_ins;                         \
    }

/*!
 * \brief register opr serialization and shallow copy methods
 */
#define MGB_SEREG_OPR_AND_REG_SHALLOW_COPY(_cls, _arity, _copy) \
    MGB_SEREG_OPR(_cls, _arity)                                 \
    MGB_REG_OPR_SHALLOW_COPY(_cls, ::mgb::serialization::_copy<_cls>)

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
