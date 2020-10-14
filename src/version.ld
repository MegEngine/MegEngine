{
global:
    MGB_VSYM_*;
    MEGDNN_VSYM_*;
    mgb_get_extern_c_opr_api_versioned;
    PyInit__imperative_rt;
    PyInit__mgb;
    extern "C++" {
        *mgb::*;
        *megdnn::*;
        *megcore::*;
        megcore*;
    };
    megcore*;

local:
    *;
};
