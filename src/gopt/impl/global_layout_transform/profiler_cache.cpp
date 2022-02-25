#include "./opr_safe_dump.h"
#include "megbrain/comp_node_env.h"
#include "megbrain/gopt/profiler.h"

using namespace mgb;
using namespace gopt;
using ReformatKey = ReformatManager::ReformatKey;

// =================== ProfilerCache ======================
void ProfilerCache::Key::build_blob_from_opr() {
    auto&& opr = m_key_impl.opr_key.opr;

    // process  opr param
    auto data = intl::opr_safe_dump(opr);
    size_t param_size = data.size();

    size_t nr_inputs = opr->input().size();
    size_t nr_outputs = opr->usable_output().size();
    size_t nr_layouts = nr_inputs + nr_outputs;
    m_blob_storage.reserve(sizeof(TensorLayout) * 3 * nr_layouts + param_size);

    // serialize param
    const char* data_ptr = reinterpret_cast<const char*>(data.data());
    m_blob_storage.append(data_ptr, param_size);

    // serialize layouts
    auto append_layout = [this](const VarNode* v) {
        TensorLayout ly{v->shape(), v->dtype(), v->format()};
        for (size_t i = 0; i < ly.ndim; ++i) {
            if (i)
                m_blob_storage.push_back(',');
            m_blob_storage.append(std::to_string(ly.shape[i]));
        }
        if (!ly.is_contiguous()) {
            m_blob_storage.push_back(';');
            for (size_t i = 0; i < ly.ndim; ++i) {
                if (i)
                    m_blob_storage.push_back(',');
                m_blob_storage.append(std::to_string(ly.stride[i]));
            }
        }
        m_blob_storage.push_back(';');
        m_blob_storage.append(ly.dtype.name());
        m_blob_storage.push_back('|');
    };
    for (size_t i = 0; i < nr_inputs; ++i) {
        append_layout(opr->input(i));
    }
    for (size_t i = 0; i < nr_outputs; ++i) {
        append_layout(opr->output(i));
    }

    // serialize opr_format
    m_blob_storage.append(
            std::to_string(static_cast<uint32_t>(m_key_impl.opr_key.config_id)));

    // serialize extra_attribute
    m_blob_storage.append(
            std::to_string(static_cast<uint32_t>(m_key_impl.opr_key.extra_attribute)));
}

void ProfilerCache::Key::build_category(CompNode cn) {
    m_category = "layout_transform_profile:";
    auto&& env = CompNodeEnv::from_comp_node(cn);
    switch (env.property().type) {
#if MGB_CUDA
        case CompNode::DeviceType::CUDA: {
            m_category += "plat=cuda";
            if (ProfilerCache::inst().enable_device_info()) {
                auto&& prop = env.cuda_env().device_prop;
                m_category += ssprintf(
                        ";dev=%s;cap=%d.%d", prop.name, prop.major, prop.minor);
            }
            break;
        }
#endif
        case CompNode::DeviceType::CPU:
            m_category += "plat=cpu";
            break;
        default:
            mgb_throw(
                    MegBrainError,
                    "unsupported comp node for global layout transform "
                    "profiler cache category");
    }
}

void ProfilerCache::Key::build_blob_from_var() {
    auto v = m_key_impl.var_key.var;

    // serialize layouts
    auto append_layout = [this](const VarNode* v) {
        TensorLayout ly{v->shape(), v->dtype(), v->format()};
        for (size_t i = 0; i < ly.ndim; ++i) {
            if (i)
                m_blob_storage.push_back(',');
            m_blob_storage.append(std::to_string(ly.shape[i]));
        }
        if (!ly.is_contiguous()) {
            m_blob_storage.push_back(';');
            for (size_t i = 0; i < ly.ndim; ++i) {
                if (i)
                    m_blob_storage.push_back(',');
                m_blob_storage.append(std::to_string(ly.stride[i]));
            }
        }
        m_blob_storage.push_back(';');
        m_blob_storage.append(ly.dtype.name());
        m_blob_storage.push_back('|');
    };
    append_layout(v);

    // serialze reformat key
    m_blob_storage.append(m_key_impl.var_key.key.to_string());
}

const std::string& ProfilerCache::Key::category() const {
    mgb_assert(!m_category.empty());
    return m_category;
}

PersistentCache::Blob ProfilerCache::Key::blob() const {
    mgb_assert(!m_blob_storage.empty());
    return {m_blob_storage.data(), m_blob_storage.size()};
}

ProfilerCache& ProfilerCache::inst() {
    static ProfilerCache inst;
    return inst;
}

ProfilerCache& ProfilerCache::set_impl(std::unique_ptr<PersistentCache> impl) {
    mgb_assert(impl != nullptr);
    m_impl.swap(impl);
    return *this;
}

void ProfilerCache::dump_cache(const char* path) {
    mgb_assert(
            m_impl->support_dump_cache(),
            "current impl of ProfilerCache does not support dump cache to "
            "file.");
    auto cache = static_cast<InFilePersistentCache*>(m_impl.get());
    cache->dump_cache(path);
}

Maybe<ProfilerCache::Result> ProfilerCache::get(const Key& key) {
    auto raw_buf = m_impl->get(key.category(), key.blob());
    if (!raw_buf.valid())
        return None;
    // data type of cost is float
    auto buf = static_cast<const uint8_t*>(raw_buf->ptr);
    auto size = raw_buf->size;
    mgb_assert(
            buf && size == sizeof(float),
            "ProfileCache invalid value: ptr=%p, size=%zu", buf, size);
    auto read_f32 = [&]() {
        auto ret = *reinterpret_cast<const float*>(buf);
        return ret;
    };
    auto cost = read_f32();
    return cost;
}

void ProfilerCache::put(const Key& key, Result& result) {
    std::string val;
    megdnn::Algorithm::serialize_write_pod(result, val);
    m_impl->put(key.category(), key.blob(), {val.data(), val.size()});
}

// vim: syntax=cpp.doxygen
