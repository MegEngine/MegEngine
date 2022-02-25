#ifndef LITE_MACRO_H_
#define LITE_MACRO_H_

#if defined(_WIN32)
#define LITE_API __declspec(dllexport)
#else
#define LITE_API __attribute__((visibility("default")))
#endif
#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
