namespace megdnn {
namespace cuda {
namespace mask_conv {

template <typename ctype>
void set_zero_by_mask_proxy(
        float* dst, const ctype* mask, size_t N, size_t OC, size_t OH, size_t OW,
        cudaStream_t stream);

template <typename ctype>
void mask_propagate_exec_proxy(
        const ctype* src, ctype* dst, size_t IH, size_t IW, size_t OH, size_t OW,
        size_t FH, size_t FW, size_t SH, size_t SW, size_t PH, size_t PW, size_t DH,
        size_t DW, cudaStream_t stream);

}  // namespace mask_conv

}  // namespace cuda
}  // namespace megdnn
