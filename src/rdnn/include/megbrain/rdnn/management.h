#pragma once

#include "megbrain/comp_node.h"
#include "megdnn/handle.h"

namespace mgb {
namespace opr {
namespace intl {

//! get megdnn handle from comp node
MGE_WIN_DECLSPEC_FUC megdnn::Handle* get_megdnn_handle(CompNode comp_node);
MGE_WIN_DECLSPEC_FUC std::shared_ptr<megdnn::Handle> get_megdnn_handle_shared(
        CompNode comp_node);

/*!
 * \brief get global megdnn operator asscoated with a computing node
 * \tparam Opr megdnn operator class, must be one of:
 *      * AddUpdate
 *      * Relayout
 *      * Checksum
 */
template <typename Opr>
MGE_WIN_DECLSPEC_FUC Opr* get_megdnn_global_opr(CompNode comp_node);

template <class Obj>
class UniqPtrWithCN : public std::unique_ptr<Obj> {
    CompNode m_cn;

public:
    UniqPtrWithCN() = default;

    template <class RObj>
    UniqPtrWithCN(UniqPtrWithCN<RObj>&& o)
            : std::unique_ptr<Obj>(std::move(o)), m_cn(o.comp_node()) {}

    UniqPtrWithCN(std::unique_ptr<Obj> ptr, CompNode cn)
            : std::unique_ptr<Obj>{std::move(ptr)}, m_cn{cn} {}

    CompNode comp_node() const { return m_cn; }
};

//! create megdnn opr from megdnn handle in a CompNode
template <class Opr>
UniqPtrWithCN<Opr> create_megdnn_opr(CompNode comp_node) {
    return {get_megdnn_handle(comp_node)->create_operator<Opr>(), comp_node};
}

}  // namespace intl
}  // namespace opr

namespace rdnn {
template <typename Obj>
using UniqPtrWithCN = opr::intl::UniqPtrWithCN<Obj>;
}  // namespace rdnn
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
