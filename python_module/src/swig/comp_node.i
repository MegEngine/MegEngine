/*
 * $File: comp_node.i
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * $Copyright: Copyright (c) 2014-2017 Megvii Inc. All rights reserved.
 */

%{
using mgb::CompNode;
static CompNode::DeviceType str2device_type(
        const std::string &str, bool allow_unspec) {
    using T = CompNode::DeviceType;
    if (str == "CPU") {
        return T::CPU;
    } else if (str == "CUDA" || str == "GPU") {
        return T::CUDA;
    } else {
        mgb_assert(allow_unspec && str == "XPU", "bad device type: %s; which "
                "must be either CPU, GPU or XPU", str.c_str());
        return T::UNSPEC;
    }
}
%}

class CompNode {
    public:
        static CompNode load(const char* id);

        %extend {
            static void _set_device_map(const std::string &type,
                    int from, int to) {
                CompNode::Locator::set_device_map(
                        str2device_type(type, false), from, to);
            }

            static size_t _get_device_count(const std::string &type, bool warn) {
                return CompNode::get_device_count(str2device_type(type, true), warn);
            }

            static void _set_unspec_device_type(const std::string &type) {
                CompNode::Locator::set_unspec_device_type(
                    str2device_type(type, false));
            }

            bool _check_eq(const CompNode &rhs) const {
                return (*$self) == rhs;
            }

            std::vector<int> _get_locator() const {
                auto logi = $self->locator_logical(), phys = $self->locator();
                return {
                    static_cast<int>(logi.type), logi.device, logi.stream,
                            static_cast<int>(phys.type), phys.device,
                            phys.stream,
                };
            }

            std::string __getstate__() {
                return $self->to_string_logical();
            }

            std::string __str__() {
                return $self->to_string();
            }

            std::string __repr__() {
                return mgb::ssprintf("CompNode(\"%s\" from \"%s\")",
                        $self->to_string().c_str(),
                        $self->to_string_logical().c_str());
            }

            size_t _get_mem_align_() const {
                return $self->get_mem_addr_alignment();
            }

            size_t __hash__() {
                return mgb::hash(*$self);
            }
        }

        %pythoncode {
            DEVICE_TYPE_MAP = {
                0: 'XPU',
                1: 'CUDA',
                2: 'CPU'
            }

            def __setstate__(self, state):
                self.this = CompNode_load(state).this

            def __eq__(self, rhs):
                return isinstance(rhs, CompNode) and self._check_eq(rhs)

            @property
            def mem_align(self):
                """memory alignment in bytes"""
                return self._get_mem_align_()

            @property
            def locator_logical(self) -> [str, int, int]:
                """logical locator: a tuple containing (type, device, stream)"""
                t, d, s = self._get_locator()[:3]
                return self.DEVICE_TYPE_MAP[t], d, s

            @property
            def locator_physical(self) -> [str, int, int]:
                """physical locator: a tuple containing (type, device, stream)"""
                t, d, s = self._get_locator()[3:]
                return self.DEVICE_TYPE_MAP[t], d, s
        }
};
%template(_VectorCompNode) std::vector<CompNode>;
%template(_VectorCompNodeAndSize) std::vector<std::pair<CompNode, size_t>>;

%pythoncode {

def as_comp_node(desc):
    """create a :class:`.CompNode` by desc

    :type desc: str or :class:`.CompNode`
    :param desc: if str, an id describing the comp node, like 'gpu0', 'gpu1'. A
        special id 'gpux' represents the logical default comp node. Otherwise
        it should already be a :class:`.CompNode`.
    """
    if isinstance(desc, str):
        return CompNode_load(desc)
    assert isinstance(desc, CompNode), (
        'could not convert {} to CompNode'.format(desc))
    return desc

}

// vim: ft=swig
