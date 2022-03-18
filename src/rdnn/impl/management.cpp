#include "megbrain/rdnn/management.h"
#include "megbrain/comp_node_env.h"
#include "megbrain/tensor.h"
#include "megbrain/utils/metahelper.h"

#include "megdnn/handle.h"
#include "megdnn/oprs.h"

/* ================== global functions ================== */

using namespace mgb;
using namespace mgb::opr;

namespace {
template <class Opr>
class MegDNNGlobalOprContainer final : public UserDataContainer::UserData {
    MGB_TYPEINFO_OBJ_DECL;

    std::shared_ptr<megdnn::Handle> m_megdnn_handle;
    std::unique_ptr<Opr> m_opr;

public:
    MegDNNGlobalOprContainer(CompNode cn)
            : m_megdnn_handle{intl::get_megdnn_handle_shared(cn)},
              m_opr{m_megdnn_handle->create_operator<Opr>()} {
        mgb_assert(m_opr->is_thread_safe());
    }

    Opr* get() const { return m_opr.get(); }
};

template <class Opr>
MGB_TYPEINFO_OBJ_IMPL(MegDNNGlobalOprContainer<Opr>);
}  // anonymous namespace

std::shared_ptr<megdnn::Handle> intl::get_megdnn_handle_shared(CompNode comp_node) {
    auto& handle = MegDNNHandle::get(CompNodeEnv::from_comp_node(comp_node));
    return {handle.shared_from_this(), handle.handle()};
}

megdnn::Handle* intl::get_megdnn_handle(CompNode comp_node) {
    return MegDNNHandle::get(CompNodeEnv::from_comp_node(comp_node)).handle();
}

template <typename Opr>
Opr* intl::get_megdnn_global_opr(CompNode comp_node) {
    using T = MegDNNGlobalOprContainer<Opr>;
    auto maker = [comp_node]() { return std::make_shared<T>(comp_node); };
    return CompNodeEnv::from_comp_node(comp_node).get_user_data<T>(maker).get();
}

namespace mgb {
namespace opr {
namespace intl {

#define INST(o) template o* get_megdnn_global_opr<o>(CompNode)
INST(megdnn::AddUpdate);
INST(megdnn::Relayout);
INST(megdnn::Checksum);
#undef INST

}  // namespace intl
}  // namespace opr
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
