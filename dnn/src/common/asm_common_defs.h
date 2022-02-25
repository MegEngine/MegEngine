#pragma once

#if defined(__WIN32__) || defined(__APPLE__)
#define cdecl(s) _##s
#else
#define cdecl(s) s
#endif

#if !defined(__APPLE__)
#define hidden_sym(s) .hidden cdecl(s)
#else
#define hidden_sym(s) .private_extern cdecl(s)
#endif

#if defined(__linux__) && defined(__ELF__) && (defined(__arm__) || defined(__aarch64__))
.pushsection.note.GNU - stack, "", % progbits.popsection
#endif
