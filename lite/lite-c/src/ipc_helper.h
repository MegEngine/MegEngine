#pragma once

#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>

#include "ipc_imp.h"

#define IPC_INSTACE() ipc::IpcHelper::Instance()

#define IPC_HELP_REMOTE_CALL(SHM_PTR, REMOTEFUNCID)         \
    struct ipc_imp::MsgBody msg;                            \
    msg.type = ipc_imp::IPC_CALL_REMOTE_API;                \
    msg.shm_ptr = static_cast<void*>(SHM_PTR);              \
    msg.remote_func_id = static_cast<size_t>(REMOTEFUNCID); \
    IPC_INSTACE().send_ipc_msg(&msg);

#define ASSERT_SHM_SIZE(SHM_SIZE, NEED_SIZE)                                        \
    do {                                                                            \
        if (SHM_SIZE < NEED_SIZE) {                                                 \
            LITE_ERROR(                                                             \
                    "shm_size not enough to run this api need vs config: (%fMB "    \
                    "%fMB), please config it by env: LITE_DEBUG_IPC_SHM_SIZE, for " \
                    "example config to 20MB by: export LITE_DEBUG_IPC_SHM_SIZE=20", \
                    NEED_SIZE / 1024.0 / 1024.0, SHM_SIZE / 1024.0 / 1024.0);       \
            __builtin_trap();                                                       \
        }                                                                           \
    } while (0)

namespace ipc {

template <class T>
class Singleton {
public:
    Singleton() {}
    static T& Instance() {
        static T _;
        return _;
    }
};

class IpcHelper : public Singleton<IpcHelper> {
public:
    IpcHelper(const IpcHelper&) = delete;
    IpcHelper& operator=(const IpcHelper&) = delete;
    IpcHelper();

    ~IpcHelper();

    //! send msg with default timeout
    struct ipc_imp::MsgBody send_ipc_msg(struct ipc_imp::MsgBody* msg) {
        return send_msg(msg, &tv);
    }

    //! get shm ptr
    void* get_shm_ptr(void* consumer_ptr);

    //! release shm_ptr, NOT free shm_ptr
    void release_shm_ptr(void* consumer_ptr);

    //! check shm size
    void check_shm_size(size_t need_size) { ASSERT_SHM_SIZE(shm_size, need_size); }

    //! is enable ipc fork debug mode
    static bool is_enable_fork_debug_mode() { return sm_is_enable_fork_ipc; }

    static bool sm_is_enable_fork_ipc;

private:
    //! 5 minutes
    struct timeval tv = {300, 0};

    //! map of <shm_ptr, consumer_ptr>,
    std::map<void*, void*> m_shm_ptr2consumer_ptr;

    size_t shm_size = 0;

    //! shm_mem for consumer_ptr == nullptr
    void* shm_mem_for_null_consumer_ptr;

    LITE_MUTEX m_mtx;
};

enum class RemoteFuncId : size_t {
    LITE_MAKE_NETWORK = 1,
    LITE_LOAD_MODEL_FROM_PATH = 2,
    LITE_GET_LAST_ERROR = 3,
    LITE_GET_IO_TENSOR = 4,
    LITE_GET_TENSOR_TOTAL_SIZE_IN_BYTE = 5,
    LITE_GET_TENSOR_MEMORY = 6,
    LITE_MEMSET = 7,
    LITE_FORWARD = 8,
    LITE_WAIT = 9,
    LITE_GET_OUTPUT_NAME = 10,
    LITE_COPY_SERVER_TENSOR_MEMORY = 11,
    LITE_DESTROY_NETWORK = 12,
};

}  // namespace ipc
