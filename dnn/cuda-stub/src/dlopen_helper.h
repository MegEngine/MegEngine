#if defined(_WIN32)
#include <windows.h>
#define RTLD_LAZY 0

static void* dlopen(const char* file, int) {
    return static_cast<void*>(LoadLibraryA(file));
}

static void* dlerror() {
    const char* errmsg = "dlerror not aviable in windows";
    return const_cast<char*>(errmsg);
}

static void* dlsym(void* handle, const char* name) {
    FARPROC symbol = GetProcAddress((HMODULE)handle, name);
    return reinterpret_cast<void*>(symbol);
}

#else
#include <dlfcn.h>
#include <unistd.h>
#endif

#include <sstream>
#include <string>
#include <vector>
static std::vector<std::string> split_string(const std::string& s, char delim) {
    std::vector<std::string> elems;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

static std::vector<std::string> get_env_dir(const char* env_name) {
    const char* env_p = std::getenv(env_name);
    std::vector<std::string> env_dir;
    if (env_p) {
        env_dir = split_string(env_p, ':');
    }
    return env_dir;
}

static void* try_open_handle(std::vector<std::string> dir_vec,
                             std::string default_so_name) {
    void* handle = nullptr;
    for (auto& tk_path : dir_vec) {
        handle = dlopen((tk_path + "/" + default_so_name).c_str(), RTLD_LAZY);
        if (handle) {
            break;
        }
    }
    return handle;
}

static void* try_open_handle(const char** so_vec, int nr_so) {
    void* handle = nullptr;
    for (int i = 0; i < nr_so; ++i) {
        handle = dlopen(so_vec[i], RTLD_LAZY);
        if (handle) {
            break;
        }
    }
    return handle;
}

static void* get_library_handle() {
    std::vector<std::string> cuda_tk_dir = get_env_dir("CUDA_TK_PATH");
    std::vector<std::string> ld_dir = get_env_dir("LD_LIBRARY_PATH");
    void* handle = nullptr;
    if (!handle) {
        handle = try_open_handle(ld_dir, default_so_name);
    }
    if (!handle) {
        handle = try_open_handle(cuda_tk_dir, default_so_name);
    }
    if (!handle) {
        handle = try_open_handle(default_so_paths,
                                 sizeof(default_so_paths) / sizeof(char*));
    }
    if (!handle) {
        handle = try_open_handle(extra_so_paths,
                                 sizeof(extra_so_paths) / sizeof(char*));
    }
    if (!handle) {
        LOGE("Failed to load %s API library", g_default_api_name);
        return nullptr;
    }
    return handle;
}

static void log_failed_load(int func_idx) {
    LOGE("failed to load %s func: %s", g_default_api_name,
         g_func_name[func_idx]);
}

static void* resolve_library_func(void* handle, const char* func) {
    if (!handle) {
        LOGE("%s handle should not be nullptr!", g_default_api_name);
        return nullptr;
    }
    auto ret = dlsym(handle, func);
    if (!ret) {
        LOGE("failed to load %s func: %s", g_default_api_name, func);
    }
    return ret;
}
