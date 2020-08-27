#include <atomic>

#include "megbrain/imperative/op_def.h"

namespace mgb::imperative::interpreter {

struct Interpreter {
    using Handle = void*;

    struct Channel {
        virtual ~Channel() = default;

        virtual Handle put(const HostTensorND& value) = 0;

        virtual void del(Handle) = 0;

        virtual SmallVector<Handle> apply_op(
                std::shared_ptr<OpDef> op,
                const SmallVector<Handle>& inputs) = 0;

        virtual HostTensorND get_value(Handle) = 0;
        virtual TensorShape get_shape(Handle) = 0;
        virtual DType get_dtype(Handle) = 0;
        virtual CompNode get_device(Handle) = 0;

        virtual DeviceTensorND get_dev_tensor(Handle) = 0;

        virtual void sync() = 0;
        virtual void close() = 0;

        virtual void config_async_level(int level) = 0;
    };

    virtual std::unique_ptr<Channel> create_channel() = 0;

    static Interpreter& inst();
};

} // namespace mgb::imperative::interpreter
