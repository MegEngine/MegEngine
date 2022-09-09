#include "lite/tensor.h"
#include <set>
#include <string>
#include <unordered_map>
#include "../../src/tensor_impl_base.h"
#include "common.h"
#include "ipc_helper.h"
#include "lite-c/tensor_c.h"

const LiteLayout default_layout = {
        .shapes = {0, 0, 0, 0, 0}, .ndim = 0, .data_type = LiteDataType::LITE_FLOAT};

const LiteTensorDesc default_desc = {
        .is_pinned_host = false,
        .layout = default_layout,
        .device_type = LiteDeviceType::LITE_CPU,
        .device_id = 0};
namespace {

static LITE_MUTEX mtx_tensor;
std::unordered_map<void*, std::shared_ptr<lite::Tensor>>& get_global_tensor_holder() {
    static std::unordered_map<void*, std::shared_ptr<lite::Tensor>> global_holder;
    return global_holder;
}

static LITE_MUTEX mtx_attr;
std::unordered_map<std::string, lite::LiteAny>& get_global_tensor_attr_holder() {
    static std::unordered_map<std::string, lite::LiteAny> global_holder;
    return global_holder;
}
}  // namespace

//! convert the lite::Layout to Layout
LiteLayout convert_to_clayout(const lite::Layout& layout) {
    LiteLayout clayout;
    clayout.ndim = layout.ndim;
    LITE_ASSERT(layout.ndim < LAYOUT_MAX_DIM, "layout ndim is to large");
    for (size_t i = 0; i < layout.ndim; i++) {
        clayout.shapes[i] = layout.shapes[i];
    }
    clayout.data_type = layout.data_type;
    return clayout;
}

//! convert the C Layout to lite::Layout
lite::Layout convert_to_layout(const LiteLayout& clayout) {
    lite::Layout layout;
    layout.ndim = clayout.ndim;
    LITE_ASSERT(layout.ndim < LAYOUT_MAX_DIM, "clayout ndim is to large");
    for (size_t i = 0; i < layout.ndim; i++) {
        layout.shapes[i] = clayout.shapes[i];
    }
    layout.data_type = clayout.data_type;
    return layout;
}

int LITE_make_tensor(const LiteTensorDesc tensor_describe, LiteTensor* tensor) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(tensor, "The tensor pass to LITE_make_tensor is null");
    lite::Layout layout = convert_to_layout(tensor_describe.layout);
    auto lite_tensor = std::make_shared<lite::Tensor>(
            tensor_describe.device_id, tensor_describe.device_type, layout,
            tensor_describe.is_pinned_host);
    {
        LITE_LOCK_GUARD(mtx_tensor);
        get_global_tensor_holder()[lite_tensor.get()] = lite_tensor;
    }
    *tensor = lite_tensor.get();
    LITE_CAPI_END();
}

int LITE_destroy_tensor(LiteTensor tensor) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(tensor, "The tensor pass to LITE c_api is null");
    LITE_LOCK_GUARD(mtx_tensor);
    auto& global_holder = get_global_tensor_holder();
    if (global_holder.find(tensor) != global_holder.end()) {
        global_holder.erase(tensor);
    }
    LITE_CAPI_END();
}

int LITE_set_tensor_layout(LiteTensor tensor, const LiteLayout layout) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(tensor, "The tensor pass to LITE c_api is null");
    auto tensor_ptr = static_cast<lite::Tensor*>(tensor);
    tensor_ptr->set_layout(convert_to_layout(layout));
    LITE_CAPI_END();
}

int LITE_reset_tensor_memory(
        LiteTensor tensor, void* prepared_data, size_t data_length_in_byte) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(tensor, "The tensor pass to LITE c_api is null");
    LITE_ASSERT(prepared_data, "The prepared_data pass to LITE c_api is null");
    static_cast<lite::Tensor*>(tensor)->reset(prepared_data, data_length_in_byte);
    LITE_CAPI_END();
}

int LITE_reset_tensor(LiteTensor tensor, const LiteLayout layout, void* prepared_data) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(tensor, "The tensor pass to LITE c_api is null");
    LITE_ASSERT(prepared_data, "The prepared_data pass to LITE c_api is null");
    static_cast<lite::Tensor*>(tensor)->reset(prepared_data, convert_to_layout(layout));
    LITE_CAPI_END();
}

int LITE_tensor_reshape(LiteTensor tensor, const int* shape, int size) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(tensor && shape, "The tensor pass to LITE c_api is null");
    std::vector<int> shapes;
    for (int i = 0; i < size; i++) {
        shapes.push_back(shape[i]);
    }
    static_cast<lite::Tensor*>(tensor)->reshape(shapes);
    LITE_CAPI_END();
}

int LITE_tensor_slice(
        const LiteTensor tensor, const size_t* start, const size_t* end,
        const size_t* step, size_t size, LiteTensor* slice_tensor) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(
            tensor && start && end && slice_tensor,
            "The tensor pass to LITE c_api is null");
    std::vector<size_t> starts, ends, steps;
    for (size_t i = 0; i < size; i++) {
        starts.push_back(start[i]);
        ends.push_back(end[i]);
        if (step) {
            steps.push_back(step[i]);
        }
    }
    auto ret_tensor = static_cast<lite::Tensor*>(tensor)->slice(starts, ends, steps);
    {
        LITE_LOCK_GUARD(mtx_tensor);
        get_global_tensor_holder()[ret_tensor.get()] = ret_tensor;
    }
    *slice_tensor = ret_tensor.get();
    LITE_CAPI_END();
}

int LITE_tensor_fill_zero(LiteTensor tensor) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(tensor, "The tensor pass to LITE c_api is null");
    static_cast<lite::Tensor*>(tensor)->fill_zero();
    LITE_CAPI_END();
}

int LITE_tensor_copy(LiteTensor dst_tensor, const LiteTensor src_tensor) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(dst_tensor && src_tensor, "The tensor pass to LITE c_api is null");
    static_cast<lite::Tensor*>(dst_tensor)
            ->copy_from(*static_cast<lite::Tensor*>(src_tensor));
    LITE_CAPI_END();
}

int LITE_tensor_share_memory_with(LiteTensor dst_tensor, const LiteTensor src_tensor) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(dst_tensor && src_tensor, "The tensor pass to LITE c_api is null");
    static_cast<lite::Tensor*>(dst_tensor)
            ->share_memory_with(*static_cast<lite::Tensor*>(src_tensor));
    LITE_CAPI_END();
}

int LITE_get_tensor_memory(const LiteTensor tensor, void** data) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(tensor, "The tensor pass to LITE c_api is null");
    LITE_ASSERT(data, "The data ptr pass to LITE c_api is null");
    if (ipc_imp::is_server()) {
        *data = static_cast<lite::Tensor*>(tensor)->get_memory_ptr();
    } else {
        size_t need_size = sizeof(LiteTensor);
        IPC_INSTACE().check_shm_size(need_size);

        void* raw_shm_ptr = IPC_INSTACE().get_shm_ptr(nullptr);

        char* shm_ptr_c = static_cast<char*>(raw_shm_ptr);
        memcpy(shm_ptr_c, &tensor, sizeof(LiteTensor));

        IPC_HELP_REMOTE_CALL(raw_shm_ptr, ipc::RemoteFuncId::LITE_GET_TENSOR_MEMORY);

        int* ret_ptr = static_cast<int*>(raw_shm_ptr);
        auto ret = *ret_ptr;
        ret_ptr++;
        memcpy(data, ret_ptr, sizeof(void*));
        return ret;
    }
    LITE_CAPI_END();
}

void* LITE_memset(void* s, int c, size_t n) {
    if (ipc_imp::is_server()) {
        return memset(s, c, n);
    } else {
        size_t need_size = sizeof(void*) + sizeof(int) + sizeof(size_t);
        IPC_INSTACE().check_shm_size(need_size);

        void* raw_shm_ptr = IPC_INSTACE().get_shm_ptr(nullptr);

        char* shm_ptr_c = static_cast<char*>(raw_shm_ptr);
        memcpy(shm_ptr_c, &s, sizeof(void*));
        memcpy(shm_ptr_c + sizeof(void*), &c, sizeof(int));
        memcpy(shm_ptr_c + sizeof(void*) + sizeof(int), &n, sizeof(size_t));

        IPC_HELP_REMOTE_CALL(raw_shm_ptr, ipc::RemoteFuncId::LITE_MEMSET);

        return s;
    }
}

int LITE_copy_server_tensor_memory(
        void* server_ptr, void* client_ptr, size_t size_in_byte) {
    LITE_CAPI_BEGIN();
    if (ipc_imp::is_server()) {
        LITE_ASSERT(
                false, "lite not in fork debug mode, please do not call this function");
    } else {
        size_t need_size = sizeof(void*) + sizeof(size_t);
        IPC_INSTACE().check_shm_size(need_size);
        IPC_INSTACE().check_shm_size(size_in_byte);

        void* raw_shm_ptr = IPC_INSTACE().get_shm_ptr(nullptr);

        char* shm_ptr_c = static_cast<char*>(raw_shm_ptr);
        memcpy(shm_ptr_c, &server_ptr, sizeof(void*));
        memcpy(shm_ptr_c + sizeof(void*), &size_in_byte, sizeof(size_t));

        IPC_HELP_REMOTE_CALL(
                raw_shm_ptr, ipc::RemoteFuncId::LITE_COPY_SERVER_TENSOR_MEMORY);
        memcpy(client_ptr, raw_shm_ptr, size_in_byte);
        return 0;
    }
    LITE_CAPI_END();
}

int LITE_get_tensor_memory_with_index(
        const LiteTensor tensor, const size_t* index, size_t size, void** data) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(tensor && index && data, "The tensor pass to LITE c_api is null");
    std::vector<size_t> index_v;
    for (size_t i = 0; i < size; i++) {
        index_v.push_back(index[i]);
    }
    *data = static_cast<lite::Tensor*>(tensor)->get_memory_ptr(index_v);
    LITE_CAPI_END();
}

int LITE_get_tensor_total_size_in_byte(const LiteTensor tensor, size_t* size) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(tensor, "The tensor pass to LITE c_api is null");
    LITE_ASSERT(size, "The size ptr pass to LITE c_api is null");
    if (ipc_imp::is_server()) {
        *size = static_cast<lite::Tensor*>(tensor)->get_tensor_total_size_in_byte();
    } else {
        size_t need_size = sizeof(LiteTensor);
        IPC_INSTACE().check_shm_size(need_size);

        void* raw_shm_ptr = IPC_INSTACE().get_shm_ptr(nullptr);

        char* shm_ptr_c = static_cast<char*>(raw_shm_ptr);
        memcpy(shm_ptr_c, &tensor, sizeof(LiteTensor));

        IPC_HELP_REMOTE_CALL(
                raw_shm_ptr, ipc::RemoteFuncId::LITE_GET_TENSOR_TOTAL_SIZE_IN_BYTE);

        int* ret_ptr = static_cast<int*>(raw_shm_ptr);
        auto ret = *ret_ptr;
        ret_ptr++;
        memcpy(size, ret_ptr, sizeof(size_t));
        return ret;
    }
    LITE_CAPI_END();
}

int LITE_get_tensor_layout(const LiteTensor tensor, LiteLayout* layout) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(tensor, "The tensor pass to LITE c_api is null");
    LITE_ASSERT(layout, "The layout ptr pass to LITE c_api is null");
    *layout = convert_to_clayout(static_cast<lite::Tensor*>(tensor)->get_layout());
    LITE_CAPI_END();
}

int LITE_get_tensor_device_type(const LiteTensor tensor, LiteDeviceType* device_type) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(tensor, "The tensor pass to LITE c_api is null");
    LITE_ASSERT(device_type, "The device ptr pass to LITE c_api is null");
    *device_type = static_cast<lite::Tensor*>(tensor)->get_device_type();
    LITE_CAPI_END();
}

int LITE_get_tensor_device_id(const LiteTensor tensor, int* device_id) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(tensor && device_id, "The tensor pass to LITE c_api is null");
    *device_id = static_cast<lite::Tensor*>(tensor)->get_device_id();
    LITE_CAPI_END();
}

int LITE_is_pinned_host(const LiteTensor tensor, int* is_pinned_host) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(tensor, "The tensor pass to LITE c_api is null");
    LITE_ASSERT(is_pinned_host, "The is_pinned_host ptr pass to LITE c_api is null");
    *is_pinned_host = static_cast<lite::Tensor*>(tensor)->is_pinned_host();
    LITE_CAPI_END();
}

int LITE_is_memory_continue(const LiteTensor tensor, int* is_continue) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(tensor, "The tensor pass to LITE c_api is null");
    LITE_ASSERT(is_continue, "The is_continue ptr pass to LITE c_api is null");
    *is_continue = static_cast<lite::Tensor*>(tensor)->is_continue_memory();
    LITE_CAPI_END();
}

int LITE_tensor_concat(
        LiteTensor* tensors, int nr_tensor, int dim, LiteDeviceType dst_device,
        int device_id, LiteTensor* result_tensor) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(result_tensor, "The tensor pass to LITE c_api is null");
    std::vector<lite::Tensor> v_tensors;
    for (int i = 0; i < nr_tensor; i++) {
        v_tensors.push_back(*static_cast<lite::Tensor*>(tensors[i]));
    }
    auto tensor = lite::TensorUtils::concat(v_tensors, dim, dst_device, device_id);
    {
        LITE_LOCK_GUARD(mtx_tensor);
        get_global_tensor_holder()[tensor.get()] = tensor;
    }
    *result_tensor = tensor.get();
    LITE_CAPI_END()
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
