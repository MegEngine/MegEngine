#include "ipc_helper.h"

#include "lite-c/global_c.h"
#include "lite-c/network_c.h"

#include "misc.h"

using namespace ipc_imp;
namespace ipc {

static void api_remote_call_cb(struct MsgBody* msg) {
    LITE_DEBUG(
            "into %s: %d remote_func_id: %zu", __func__, __LINE__, msg->remote_func_id);
    switch (static_cast<RemoteFuncId>(msg->remote_func_id)) {
        case RemoteFuncId::LITE_MAKE_NETWORK: {
            LiteNetwork network;
            //! second args is const LiteConfig config
            char* shm_ptr_c = static_cast<char*>(msg->shm_ptr);
            LiteConfig config;
            memcpy(&config, shm_ptr_c, sizeof(LiteConfig));
            //! third args is network_io
            LiteNetworkIO network_io;
            memcpy(&network_io, shm_ptr_c + sizeof(LiteConfig), sizeof(LiteNetworkIO));

            int ret = LITE_make_network(&network, config, network_io);

            //! API is block, put ret to shm_ptr
            int* ret_ptr = static_cast<int*>(msg->shm_ptr);
            *ret_ptr = ret;
            ret_ptr++;
            void* network_p = static_cast<void*>(ret_ptr);
            memcpy(network_p, &network, sizeof(LiteNetwork));
        }; break;
        case RemoteFuncId::LITE_LOAD_MODEL_FROM_PATH: {
            LiteNetwork network;
            char* shm_ptr_c = static_cast<char*>(msg->shm_ptr);
            memcpy(&network, shm_ptr_c, sizeof(LiteNetwork));

            int ret =
                    LITE_load_model_from_path(network, shm_ptr_c + sizeof(LiteNetwork));

            //! API is block, put ret to shm_ptr
            int* ret_ptr = static_cast<int*>(msg->shm_ptr);
            *ret_ptr = ret;
        }; break;
        case RemoteFuncId::LITE_GET_LAST_ERROR: {
            auto shm_size = base_get_shm_size();

            const char* ret = LITE_get_last_error();
            char* ret_ptr = static_cast<char*>(msg->shm_ptr);

            auto last_error_str_len = strlen(ret) + 1;
            ASSERT_SHM_SIZE(shm_size, last_error_str_len);
            strcpy(ret_ptr, ret);
        }; break;
        case RemoteFuncId::LITE_GET_IO_TENSOR: {
            char* shm_ptr_c = static_cast<char*>(msg->shm_ptr);
            LiteNetwork network;
            LiteTensorPhase phase;
            LiteTensor tensor;
            memcpy(&network, shm_ptr_c, sizeof(LiteNetwork));
            memcpy(&phase, shm_ptr_c + sizeof(LiteNetwork), sizeof(LiteTensorPhase));

            int ret = LITE_get_io_tensor(
                    network, shm_ptr_c + sizeof(LiteNetwork) + sizeof(LiteTensorPhase),
                    phase, &tensor);

            //! API is block, put ret to shm_ptr
            int* ret_ptr = static_cast<int*>(msg->shm_ptr);
            *ret_ptr = ret;
            ret_ptr++;
            void* lite_tensor_p = static_cast<void*>(ret_ptr);
            memcpy(lite_tensor_p, &tensor, sizeof(LiteTensor));
        }; break;
        case RemoteFuncId::LITE_GET_TENSOR_TOTAL_SIZE_IN_BYTE: {
            char* shm_ptr_c = static_cast<char*>(msg->shm_ptr);
            LiteTensor tensor;
            size_t size;
            memcpy(&tensor, shm_ptr_c, sizeof(LiteTensor));

            int ret = LITE_get_tensor_total_size_in_byte(tensor, &size);

            int* ret_ptr = static_cast<int*>(msg->shm_ptr);
            *ret_ptr = ret;
            ret_ptr++;
            memcpy(ret_ptr, &size, sizeof(size_t));
        }; break;
        case RemoteFuncId::LITE_GET_TENSOR_MEMORY: {
            char* shm_ptr_c = static_cast<char*>(msg->shm_ptr);
            LiteTensor tensor;
            void* data;
            memcpy(&tensor, shm_ptr_c, sizeof(LiteTensor));

            int ret = LITE_get_tensor_memory(tensor, &data);

            int* ret_ptr = static_cast<int*>(msg->shm_ptr);
            *ret_ptr = ret;
            ret_ptr++;
            memcpy(ret_ptr, &data, sizeof(void*));
        }; break;
        case RemoteFuncId::LITE_MEMSET: {
            char* shm_ptr_c = static_cast<char*>(msg->shm_ptr);
            void* s;
            int c;
            size_t n;
            memcpy(&s, shm_ptr_c, sizeof(void*));
            memcpy(&c, shm_ptr_c + sizeof(void*), sizeof(int));
            memcpy(&n, shm_ptr_c + sizeof(void*) + sizeof(int), sizeof(size_t));
            LITE_memset(s, c, n);
        }; break;
        case RemoteFuncId::LITE_FORWARD: {
            char* shm_ptr_c = static_cast<char*>(msg->shm_ptr);
            LiteNetwork network;
            memcpy(&network, shm_ptr_c, sizeof(LiteNetwork));

            int ret = LITE_forward(network);

            int* ret_ptr = static_cast<int*>(msg->shm_ptr);
            *ret_ptr = ret;
        }; break;
        case RemoteFuncId::LITE_WAIT: {
            char* shm_ptr_c = static_cast<char*>(msg->shm_ptr);
            LiteNetwork network;
            memcpy(&network, shm_ptr_c, sizeof(LiteNetwork));

            int ret = LITE_wait(network);

            int* ret_ptr = static_cast<int*>(msg->shm_ptr);
            *ret_ptr = ret;
        }; break;
        case RemoteFuncId::LITE_GET_OUTPUT_NAME: {
            char* shm_ptr_c = static_cast<char*>(msg->shm_ptr);
            LiteNetwork network;
            size_t index;
            const char* name;
            memcpy(&network, shm_ptr_c, sizeof(LiteNetwork));
            memcpy(&index, shm_ptr_c + sizeof(LiteNetwork), sizeof(size_t));

            int ret = LITE_get_output_name(network, index, &name);
            auto output_name_len = strlen(name) + 1;
            auto shm_size = base_get_shm_size();
            ASSERT_SHM_SIZE(shm_size, output_name_len);

            int* ret_ptr = static_cast<int*>(msg->shm_ptr);
            *ret_ptr = ret;
            ret_ptr++;
            void* p = static_cast<void*>(ret_ptr);
            char* p_c = static_cast<char*>(p);
            strcpy(p_c, name);
        }; break;
        case RemoteFuncId::LITE_COPY_SERVER_TENSOR_MEMORY: {
            char* shm_ptr_c = static_cast<char*>(msg->shm_ptr);
            void* server_ptr;
            size_t size_in_byte;
            memcpy(&server_ptr, shm_ptr_c, sizeof(void*));
            memcpy(&size_in_byte, shm_ptr_c + sizeof(void*), sizeof(size_t));
            memcpy(shm_ptr_c, server_ptr, size_in_byte);
        }; break;
        case RemoteFuncId::LITE_DESTROY_NETWORK: {
            char* shm_ptr_c = static_cast<char*>(msg->shm_ptr);
            LiteNetwork network;
            memcpy(&network, shm_ptr_c, sizeof(LiteNetwork));

            int ret = LITE_destroy_network(network);

            int* ret_ptr = static_cast<int*>(msg->shm_ptr);
            *ret_ptr = ret;
        }; break;
        default:
            LITE_THROW("code issue happened: do not handle RemoteFuncId");
    }
};
bool IpcHelper::sm_is_enable_fork_ipc = false;
IpcHelper::IpcHelper() {
    if (!is_server()) {
        sm_is_enable_fork_ipc = is_enable_ipc();
        auto shm_mem_conut = base_get_shm_count();
        shm_mem_for_null_consumer_ptr = base_get_shm_ptr(0);
        LITE_ASSERT(
                shm_mem_for_null_consumer_ptr, "invalid shm_ptr: %p",
                shm_mem_for_null_consumer_ptr);

        for (size_t i = 1; i < shm_mem_conut; i++) {
            void* shm_ptr = base_get_shm_ptr(i);
            LITE_ASSERT(shm_ptr, "invalid shm_ptr: %p", shm_ptr);
            //! init consumer ptr as nullptr
            m_shm_ptr2consumer_ptr[shm_ptr] = nullptr;
        }

        shm_size = base_get_shm_size();
        register_remote_call_cb(&api_remote_call_cb);
    } else {
        LITE_THROW("code issue happened: can not use for client");
    }
};

IpcHelper::~IpcHelper() {
    struct MsgBody msg;
    msg.type = IPC_SERVER_EXIT;
    send_ipc_msg(&msg);
    join_server();
};

void* IpcHelper::get_shm_ptr(void* consumer_ptr) {
    LITE_LOCK_GUARD(m_mtx);

    if (!consumer_ptr) {
        return shm_mem_for_null_consumer_ptr;
    }

    void* ret = nullptr;
    //! try find old one
    for (auto&& i : m_shm_ptr2consumer_ptr) {
        if (consumer_ptr == i.second) {
            ret = i.first;
            break;
        }
    }

    //! if not find, try alloc a new one
    if (!ret) {
        for (auto&& i : m_shm_ptr2consumer_ptr) {
            if (nullptr == i.second) {
                i.second = consumer_ptr;
                ret = i.first;
                break;
            }
        }
    }

    if (!ret) {
        LITE_ERROR(
                "can not find any usable shm_mem, may config LITE_DEBUG_IPC_SHM_COUNT "
                "big "
                "than :%zu",
                m_shm_ptr2consumer_ptr.size() + 1);
        LITE_ASSERT(ret);
    }

    return ret;
};

void IpcHelper::release_shm_ptr(void* consumer_ptr) {
    LITE_LOCK_GUARD(m_mtx);

    LITE_ASSERT(consumer_ptr, "invalid consumer_ptr ptr");
    for (auto&& i : m_shm_ptr2consumer_ptr) {
        if (consumer_ptr == i.second) {
            //! release use of shm_mem, then other consumer can use it
            i.second = nullptr;
            return;
        }
    }

    LITE_THROW(
            "error happened!!, can not find any consumer_ptr in "
            "m_shm_ptr2consumer_ptr");
};

}  // namespace ipc
