#include "./backtrace.h"
#include <cstdint>
#include "megbrain/common.h"
#include "megbrain/imperative/transformation.h"
namespace mgb::imperative::python {

static bool enable_py_bt = false;
static bool enable_trans_bt = false;

std::unordered_map<ptrdiff_t, PyObjRefKeeper> FrameInfo::code_ref_keeper = {};

std::pair<FrameInfoPtr, int> FrameInfo::make(
        PyFrameObject* frame, FrameInfoCache* cache) {
    if (frame == NULL) {
        return std::make_pair(nullptr, -1);
    }
    auto* keywrapper = TraceKeyWrapper::try_cast(frame->f_trace);
    auto key = keywrapper ? keywrapper->key : -1;
#if PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION < 10
    int lineno = frame->f_lasti;
#else
    int lineno = PyFrame_GetLineNumber(frame);
#endif

    FrameInfoPtr cache_finfo;
    if (key != -1 && cache != nullptr && key < cache->size() &&
        (*cache)[key]->lineno == lineno) {
        cache_finfo = (*cache)[key];
    }
    if (cache_finfo) {
        return std::make_pair(cache_finfo, key);
    } else {
        PyCodeObject* code = frame->f_code;
        FrameInfoPtr f = std::make_shared<FrameInfo>(code, lineno);
        if (keywrapper) {
            f->scope = keywrapper->scope;
        }
        return std::make_pair(f, -1);
    }
};

std::string FrameInfo::traceback() {
    FrameInfoPtr cur = shared_from_this();
    std::list<FrameInfo*> frames;
    while (cur) {
        frames.push_front(cur.get());
        cur = cur->prev_frame;
    }
    std::string logs;
    for (auto&& f : frames) {
        auto code = py::handle((PyObject*)f->code_obj);
#if PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION < 10
        int lineno = PyCode_Addr2Line(f->code_obj, f->lineno);
#else
        int lineno = f->lineno;
#endif
        if (f->scope != "")
            logs += "scope: <" + f->scope + ">\n";
        py::object filename = py::getattr(code, "co_filename");
        logs += "File \"";
        logs += py::str(filename);
        logs += "\", line ";
        logs += std::to_string(lineno);
        logs += ", in ";
        logs += py::str(py::getattr(code, "co_name"));
        logs += '\n';
    }
    return logs;
}

#if PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION > 6
static Py_tss_t tss_key = Py_tss_NEEDS_INIT;
#else
static int tss_key;
#endif

static bool support_tss = false;

void init_backtrace_tss_key() {
#if PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION > 6
    int result = PyThread_tss_create(&tss_key);
    support_tss = result == 0;
#else
    tss_key = PyThread_create_key();
    support_tss = tss_key != -1;
#endif
}

FrameInfoCache* FrameInfoCache::get_instance() {
    mgb_assert(support_tss);
    constexpr int max_cache_size = 10;
    static FrameInfoCache caches[max_cache_size];
    static uintptr_t tid = 1;
    static std::list<std::pair<uintptr_t, FrameInfoCache*>> cache_list;
    static std::unordered_map<uintptr_t, decltype(cache_list)::iterator> kv_map;
    auto get_cache = [](uintptr_t key) {
        if (kv_map.find(key) != kv_map.end()) {
            auto it = kv_map[key];
            auto rst = it->second;
            if (it != cache_list.begin()) {
                cache_list.push_front(*it);
                cache_list.erase(it);
                kv_map[key] = cache_list.begin();
            }
            return rst;
        }
        if (cache_list.size() < max_cache_size) {
            auto* rst = &caches[key % max_cache_size];
            cache_list.emplace_front(key, rst);
            kv_map[key] = cache_list.begin();
            return rst;
        } else {
            auto it = --cache_list.end();
            auto empty_cache = *it;
            cache_list.erase(it);
            empty_cache.second->stack_cache.clear();
            cache_list.push_front(empty_cache);
            kv_map[key] = cache_list.begin();
            return empty_cache.second;
        }
    };
#if PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION > 6
    auto* id = PyThread_tss_get(&tss_key);
    if (id == NULL) {
        mgb_assert(PyThread_tss_set(&tss_key, (void*)tid) == 0);
        return get_cache(tid++);
    } else {
        auto cache_tid = (uintptr_t)id;
        return get_cache(cache_tid);
    }
#else
    auto* id = PyThread_get_key_value(tss_key);
    if (id == NULL) {
        mgb_assert(PyThread_set_key_value(tss_key, (void*)tid) == 0);
        return get_cache(tid++);
    } else {
        auto cache_tid = (uintptr_t)id;
        return get_cache(cache_tid);
    }
#endif
}

void FrameInfoCache::update_cache(
        int key,
        const SmallVector<std::pair<PyFrameObject*, FrameInfoPtr>, 100>& frames) {
    stack_cache.resize(key + frames.size() + 1);
    auto it = frames.rbegin();
    auto cur_key = key + 1;
    for (; it != frames.rend(); it++, cur_key++) {
        auto&& [frame, finfo] = *it;
        stack_cache[cur_key] = finfo;
        if (auto* key_ptr = TraceKeyWrapper::try_cast(frame->f_trace)) {
            key_ptr->key = cur_key;
        } else {
            auto* py_key = TraceKeyWrapper::make(cur_key, frame->f_trace);
            frame->f_trace = py_key;
        }
    }
}

int FrameInfoCache::get_frame_key(PyFrameObject* frame) {
    auto* key = TraceKeyWrapper::try_cast(frame->f_trace);
    if (key == nullptr) {
        return -1;
    } else {
        return key->key;
    }
}

FrameInfoPtr get_frameinfo_from_pyframe(PyFrameObject* frame) {
    auto* cache = FrameInfoCache::get_instance();
    auto&& [cur_info, key] = FrameInfo::make(frame, cache);
    auto rst = cur_info;
    SmallVector<std::pair<PyFrameObject*, FrameInfoPtr>, 100> frames;
    py::object cur_frame = py::reinterpret_borrow<py::object>((PyObject*)frame);
    while (key == -1) {
        if (((PyFrameObject*)cur_frame.ptr())->f_gen == NULL)
            frames.push_back({(PyFrameObject*)cur_frame.ptr(), cur_info});
        auto prev_frame = py::getattr(py::handle(cur_frame), "f_back");
        if (prev_frame.is_none())
            break;
        auto&& [prev_info, prev_key] =
                FrameInfo::make((PyFrameObject*)prev_frame.ptr(), cache);
        cur_info->prev_frame = prev_info;
        cur_info = prev_info, key = prev_key;
        cur_frame = std::move(prev_frame);
    }
    if (cache != nullptr)
        cache->update_cache(key, frames);
    return rst;
}

bool set_python_backtrace(bool enabled) {
    std::swap(enable_py_bt, enabled);
    return enabled;
}

bool set_transformation_backtrace(bool enabled) {
    std::swap(enable_trans_bt, enabled);
    return enabled;
}

void record_py_backtrace() {
    auto& context = Transformation::get_context();
    context.py_traceback.clear();
    FrameInfoPtr info;
    if (enable_py_bt) {
        auto frame = PyEval_GetFrame();
        info = get_frameinfo_from_pyframe(frame);
    }
    context.bt = std::make_shared<BackTraceInfo>(std::move(info));
    context.record_bt_trans_id = context.next_transformation;
    context.record_trans_bt = enable_trans_bt;
    if (enable_py_bt) {
        context.py_traceback = context.bt->py_traceback();
    }
}

void record_scope(PyFrameObject* frame, std::string scope) {
    if (enable_py_bt) {
        frame->f_trace = TraceKeyWrapper::make(-1, frame->f_trace, std::move(scope));
    }
}

std::string get_py_backtrace() {
    auto frame = PyEval_GetFrame();
    return get_frameinfo_from_pyframe(frame)->traceback();
}

}  // namespace mgb::imperative::python