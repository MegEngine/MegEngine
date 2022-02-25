#if defined(MIDOUT_GENERATED) || defined(MIDOUT_PROFILING)
#error "midout should not be enabled on x86, because current x86 implemention requires all possible inputs to be passed in midout, which is essentially impossible in production as the input spatial size is unfixed."
#endif

// vim: syntax=cpp.doxygen
