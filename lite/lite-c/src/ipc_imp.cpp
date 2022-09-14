#include "ipc_imp.h"

#if __linux__
#include <sys/prctl.h>
#include <sys/wait.h>
#endif

#ifdef __ANDROID__
#include <android/log.h>
#include <sys/system_properties.h>
#endif

namespace ipc_imp {

namespace {

//! max count if shm
#define MAX_SHM_COUNT 15
struct ServerConfig {
#ifdef _LITE_SUPPORT_IPC
    pid_t server_id;
#else
    size_t server_id;
#endif
    void* cb;
    int fd_s[2];
    int fd_c[2];
    fd_set select_s;
    fd_set select_c;
    void* shm_ptr[MAX_SHM_COUNT];
    size_t shm_mem_conut;
    //! all shm use the same shm_size
    size_t shm_size;
};

static LITE_MUTEX m_mtx;

static ServerConfig server_config;

#ifdef _LITE_SUPPORT_IPC
static size_t config_shm_memory() {
    //! default config to 10MB
    size_t shm_size = 10 * 1024 * 1024;
    //! env to config LITE_DEBUG_IPC_SHM_SIZE
    //! for example , export LITE_DEBUG_IPC_SHM_SIZE=20, means config SHM size 20MB
    if (auto env = ::std::getenv("LITE_DEBUG_IPC_SHM_SIZE"))
        shm_size = std::stoi(env) * 1024 * 1024;

#ifdef __ANDROID__
    //! special for Android prop, attention: getprop may need permission
    char buf[PROP_VALUE_MAX];
    if (__system_property_get("LITE_DEBUG_IPC_SHM_SIZE", buf) > 0) {
        shm_size = std::stoi(buf) * 1024 * 1024;
    }
#endif

    return shm_size;
}

//! FIXME: detail at create_server(), at this stage, we only support enable lite fork
//! debug by: LITE_enable_lite_ipc_debug, after issue fix, may support env:
//! LITE_ENABLE_C_API_WITH_FORK_MODE
// bool config_enable_debug_fork() {
//    //! debug off, we only support enable fork debug mode with env
//    //! LITE_ENABLE_C_API_WITH_FORK_MODE, not support api to config
//    //! as we will fork as soon as possible by __attribute__((constructor)),
//    //! user may not have to chance to call any normal api before it
//    bool ret = false;
//    //! env to config LITE_ENABLE_C_API_WITH_FORK_MODE
//    //! for example , export LITE_ENABLE_C_API_WITH_FORK_MODE=1, means enable LITE c
//    api
//    //! with fork mode
//    if (auto env = ::std::getenv("LITE_ENABLE_C_API_WITH_FORK_MODE")) {
//        if (std::stoi(env) > 0) {
//            ret = true;
//        }
//    }
//
//#ifdef __ANDROID__
//    //! special for Android prop, attention: getprop may need permission
//    char buf[PROP_VALUE_MAX];
//    if (__system_property_get("LITE_ENABLE_C_API_WITH_FORK_MODE", buf) > 0) {
//        ret = std::stoi(buf);
//        if (std::stoi(buf) > 0) {
//            ret = true;
//        }
//    }
//#endif
//
//    return ret;
//}
#endif

static bool is_enable_debug_fork = false;

//! internal recycle server when IPC_ASSERT happen
static void recycle_server() {
    static struct timeval tv = {1, 0};
    struct MsgBody msg;
    msg.type = IPC_SERVER_EXIT;
    if (server_config.server_id > 0) {
        send_msg(&msg, &tv);
    }
}

#define ipc_unlikely(v) __builtin_expect((v), 0)
#define IPC_ASSERT(expr, msg)                                                          \
    do {                                                                               \
        if (ipc_unlikely(!(expr))) {                                                   \
            LITE_ERROR("ipc fatal error: assert failed: %s with msg: %s", #expr, msg); \
            recycle_server();                                                          \
            __builtin_trap();                                                          \
        }                                                                              \
    } while (0)

#ifdef _LITE_SUPPORT_IPC
static size_t config_shm_memory_count() {
    //! default config to 2
    size_t shm_cnt = 2;
    //! env to config LITE_DEBUG_IPC_SHM_COUNT
    //! for example , export LITE_DEBUG_IPC_SHM_COUNT=8, means config SHM count 8
    if (auto env = ::std::getenv("LITE_DEBUG_IPC_SHM_COUNT"))
        shm_cnt = std::stoi(env);

#ifdef __ANDROID__
    //! special for Android prop, attention: getprop may need permission
    char buf[PROP_VALUE_MAX];
    if (__system_property_get("LITE_DEBUG_IPC_SHM_COUNT", buf) > 0) {
        shm_cnt = std::stoi(buf);
    }
#endif
    IPC_ASSERT(
            shm_cnt >= 2 && shm_cnt <= MAX_SHM_COUNT,
            "error config LITE_DEBUG_IPC_SHM_COUNT, should be range of [2, "
            "MAX_SHM_COUNT]");

    return shm_cnt;
}
#endif

#ifdef _LITE_SUPPORT_IPC
static void handle_remote_call(struct MsgBody* msg) {
    LITE_DEBUG("into %s: %d", __func__, __LINE__);
    IPC_ASSERT(
            server_config.cb,
            "handle_remote_call failed: can not find valid "
            "remote_call_cb, please call "
            "register_remote_call_cb firstly!!");
    remote_call_cb cb = (remote_call_cb)server_config.cb;
    cb(msg);
}

static void* ipc_mmap(
        void* addr, size_t length, int prot, int flags, int fd, off_t offset) {
    void* ret = mmap(addr, length, prot, flags, fd, offset);
    IPC_ASSERT(ret != MAP_FAILED, "call mmap failed");
    return ret;
}

static int ipc_munmap(void* addr, size_t length) {
    int ret = munmap(addr, length);
    IPC_ASSERT(0 == ret, "call munmap failed");
    return ret;
}
#endif

//! start server as soon as possible
//! FIXME: when use __attribute__((constructor)) on clang, will init before all
//! static class object, which will lead to flatbuffer runtime issue, env config
//! with init_priority, do not take effect on diff cpp src file on clang
// static __attribute__((constructor)) void create_server() {
void create_server() {
#ifdef _LITE_SUPPORT_IPC
    LITE_LOCK_GUARD(m_mtx);
    LITE_LOG("try to config lite fork debug model");
    if (is_enable_debug_fork)
        return;

    is_enable_debug_fork = true;
    //! is_enable_debug_fork = config_enable_debug_fork();
    //! init default server_config
    server_config.server_id = 0;

    if (!is_enable_debug_fork)
        return;

    server_config.cb = nullptr;
    server_config.shm_size = config_shm_memory();
    server_config.shm_mem_conut = config_shm_memory_count();

    for (size_t i = 0; i < server_config.shm_mem_conut; i++) {
        server_config.shm_ptr[i] = ipc_mmap(
                NULL, server_config.shm_size, PROT_READ | PROT_WRITE,
                MAP_SHARED | MAP_ANON, -1, 0);
    }

    IPC_ASSERT(-1 != pipe(server_config.fd_s), "create server pipe failed");
    IPC_ASSERT(-1 != pipe(server_config.fd_c), "create client pipe failed");

    FD_ZERO(&server_config.select_s);
    FD_ZERO(&server_config.select_c);
    //! config server and client
    FD_SET(server_config.fd_s[0], &server_config.select_s);
    FD_SET(server_config.fd_c[0], &server_config.select_c);

    server_config.server_id = fork();

    IPC_ASSERT(server_config.server_id >= 0, "call fork failed");

    if (server_config.server_id > 0) {
        LITE_LOG("create lite_ipc_server success pid is: %d", server_config.server_id);
    } else {
        std::string server_name = "lite_ipc_server";
        // TODO: __APPLE__ do not support PR_SET_NAME and PR_SET_PDEATHSIG
        // if caller crash, no have chance to send IPC_SERVER_EXIT
#if __linux__
        LITE_LOG("start server with name: %s....", server_name.c_str());
        prctl(PR_SET_NAME, (unsigned long)server_name.c_str(), 0, 0, 0);
        //! auto die if father crash
        prctl(PR_SET_PDEATHSIG, SIGKILL);
#else
        LITE_LOG("start server....");
#endif

        while (1) {
            LITE_DEBUG("lite_ipc_server wait msg now.....");
            int res =
                    select(server_config.fd_s[0] + 1, &server_config.select_s, NULL,
                           NULL, NULL);

            IPC_ASSERT(
                    res > 0,
                    "select issue happened or timeout(but we do not support timeout)");

            struct MsgBody msg;
            size_t r_size = read(server_config.fd_s[0], &msg, sizeof(msg));
            IPC_ASSERT(r_size == sizeof(msg), "broken pipe msg");

            struct MsgBody response;
            response.type = IPC_SERVER_RESPONSE;
            switch (msg.type) {
                case IPC_CALL_REMOTE_API:
                    LITE_DEBUG("handle remote call");
                    handle_remote_call(&msg);
                    break;
                case IPC_CONFIG_REMOTE_HANDLE_API:
                    LITE_DEBUG("handle register remote cb");
                    server_config.cb = msg.cb;
                    break;
                default:
                    IPC_ASSERT(IPC_SERVER_EXIT == msg.type, "code issue happened!!");
            }

            size_t w_size = write(server_config.fd_c[1], &response, sizeof(response));
            IPC_ASSERT(w_size == sizeof(response), "write pip failed");

            if (IPC_SERVER_EXIT == msg.type) {
                LITE_DEBUG("handle exit now");
                for (size_t i = 0; i < server_config.shm_mem_conut; i++) {
                    ipc_munmap(server_config.shm_ptr[i], server_config.shm_size);
                }
                exit(0);
            }
        }
    }
#else
    //! TODO: imp for Windows with CreateProcess
    server_config.server_id = 0;
    LITE_ERROR("lite do not support fork debug ipc on this PLATFORM");
    __builtin_trap();

    return;
#endif
}

}  // namespace
//////////////////////////////////////////////// api imp /////////////////////////
void register_remote_call_cb(remote_call_cb cb) {
    IPC_ASSERT(!server_config.cb, "already register remote_call_cb");
    IPC_ASSERT(cb, "invalid remote_call_cb");
    IPC_ASSERT(server_config.server_id, "register cb need server already up");

    server_config.cb = (void*)cb;
    static struct timeval tv = {5, 0};
    struct MsgBody msg;
    msg.type = IPC_CONFIG_REMOTE_HANDLE_API;
    msg.cb = (void*)cb;
    send_msg(&msg, &tv);
}

struct MsgBody send_msg(struct MsgBody* msg, struct timeval* timeout) {
#ifdef _LITE_SUPPORT_IPC
    IPC_ASSERT(server_config.server_id > 0, "server not ready");
    if (IPC_CALL_REMOTE_API == msg->type) {
        IPC_ASSERT(
                server_config.cb,
                "can not find valid remote_call_cb, please "
                "call register_remote_call_cb firstly!!");
    }

    //! send msg to server
    size_t w_size = write(server_config.fd_s[1], msg, sizeof(struct MsgBody));
    IPC_ASSERT(w_size == sizeof(struct MsgBody), "write pipe failed");

    //! now wait server response
    struct MsgBody response;
    LITE_DEBUG("wait server response");

    int res = select(
            server_config.fd_c[0] + 1, &server_config.select_c, NULL, NULL, timeout);
    if (0 == res) {
        LITE_ERROR("wait server timeout");
    }
    IPC_ASSERT(res > 0, "select issue happened or timeout");

    size_t r_size = read(server_config.fd_c[0], &response, sizeof(response));
    IPC_ASSERT(sizeof(response) == r_size, "broken pipe msg");
    IPC_ASSERT(IPC_SERVER_RESPONSE == response.type, "error server response type");

    return response;
#else
    struct MsgBody response;
    LITE_ERROR("This code should not be called");
    __builtin_trap();

    return response;
#endif
}

bool is_server() {
    return !server_config.server_id;
}

bool is_enable_ipc() {
    return is_enable_debug_fork;
}

void join_server() {
#ifdef _LITE_SUPPORT_IPC
    if (!is_enable_debug_fork)
        return;

    int ret;
    waitpid(server_config.server_id, &ret, 0);
    if (ret) {
        //! when server crash, we mark server_config.server_id to 0
        //! to prevent handle more msg, for example recycle_server
        server_config.server_id = 0;
    }
    IPC_ASSERT(
            ret == 0, "child process exit !zero, please check server log for detail");
#endif
}

void* base_get_shm_ptr(size_t index) {
    return server_config.shm_ptr[index];
}

size_t base_get_shm_count() {
    return server_config.shm_mem_conut;
}

size_t base_get_shm_size() {
    return server_config.shm_size;
}

void enable_lite_ipc_debug() {
    create_server();
}
}  // namespace ipc_imp
