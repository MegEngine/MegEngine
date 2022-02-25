#pragma once
namespace megdnn {
enum class PostprocessMode : uint8_t {
    FLOAT = 0,   ///< support all biasmode and no_nonlinemode
    NO_PROCESS,  ///< support  non bias and identity
    QUANTIZED,   ///< support  NOBIAS ,BROADCAST_CHANNEL_BIAS and relu hswish
                 ///< identify nonline mode
    ADD_BIAS,    ///< only add bias
};
}