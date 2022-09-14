#pragma once

#undef _LITE_SUPPORT_IPC
#if __linux__ || __unix__ || __APPLE__
#define _LITE_SUPPORT_IPC
#endif

#ifdef _LITE_SUPPORT_IPC
#include <sys/mman.h>
#include <sys/time.h>
#include <sys/types.h>
#endif

#include "misc.h"

namespace ipc_imp {
//! server call api ret
enum MsgType {
    IPC_SERVER_RESPONSE = 1,
    IPC_CALL_REMOTE_API = 2,
    IPC_SERVER_EXIT = 3,
    IPC_CONFIG_REMOTE_HANDLE_API = 4,
};

struct MsgBody {
    enum MsgType type;
    //! remote call handle callback
    void* cb;
    //! remote call function emum, define by user
    size_t remote_func_id;
    //! mmap region ptr
    void* shm_ptr;
};

//! block wait server return response msg
struct MsgBody send_msg(struct MsgBody* msg, struct timeval* timeout);

//! wait server exit
void join_server();

typedef void (*remote_call_cb)(struct MsgBody* msg);

//! register remote call
void register_remote_call_cb(remote_call_cb cb);

//! is server or not, server or do not enable ipc mode will return true
bool is_server();

//! is enable ipc or not
bool is_enable_ipc();

//! get shm ptr
void* base_get_shm_ptr(size_t index);

//! get shm count
size_t base_get_shm_count();

// get shm size
size_t base_get_shm_size();

// enable fork ipc debug
void enable_lite_ipc_debug();
}  // namespace ipc_imp
