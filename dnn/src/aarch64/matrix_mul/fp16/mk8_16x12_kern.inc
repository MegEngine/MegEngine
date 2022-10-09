#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#ifndef _STR
#define _STR(X) #X
#endif
#ifndef STR
#define STR(X) _STR(X)
#endif

template <>
void matmul_mk8_16x12::kern<M_BLOCK, N_BLOCK>(
        const dt_float16* packedA, const dt_float16* packedB, int K, dt_float16* out,
        int LDC, bool is_first_k) {
#define IF_M_GT(M, INSTRUC) ".if " STR(M_BLOCK) " > " #M "\n" INSTRUC ".endif\n"
#define IF_N_GT(N, INSTRUC) ".if " STR(N_BLOCK) " > " #N "\n" INSTRUC ".endif\n"
// clang-format off
#define IF_MN_GT(M, N, INSTRUC)           \
    ".if " STR(M_BLOCK) " > " #M "\n"     \
        ".if " STR(N_BLOCK) " > " #N "\n" \
            INSTRUC                       \
        ".endif\n"                        \
    ".endif\n"

    const dt_float16* a_ptr = packedA;
    const dt_float16* b_ptr = packedB;
    dt_float16* outptr0 = out;
    dt_float16* outptr1 = out + LDC;

    int oddK = (K & 1);
    K = ((K + 1) / 2) - 1;

    asm volatile(
            "cmp %w[is_first_k], #1\n"
            "beq 1f\n"
            IF_M_GT(0, "mov x1, %[outptr0]\n")
            IF_M_GT(1, "mov x2, %[outptr1]\n")
            IF_MN_GT(0, 0, "ld1 {v8.8h}, [x1], #16\n")
            IF_MN_GT(0, 1, "ld1 {v9.8h}, [x1], #16\n")
            IF_MN_GT(0, 2, "ld1 {v10.8h}, [x1], #16\n")
            IF_MN_GT(0, 3, "ld1 {v11.8h}, [x1], #16\n")
            IF_MN_GT(0, 4, "ld1 {v12.8h}, [x1], #16\n")
            IF_MN_GT(0, 5, "ld1 {v13.8h}, [x1], #16\n")
            IF_MN_GT(0, 6, "ld1 {v14.8h}, [x1], #16\n")
            IF_MN_GT(0, 7, "ld1 {v15.8h}, [x1], #16\n")
            IF_MN_GT(0, 8, "ld1 {v16.8h}, [x1], #16\n")
            IF_MN_GT(0, 9, "ld1 {v17.8h}, [x1], #16\n")
            IF_MN_GT(0, 10, "ld1 {v18.8h}, [x1], #16\n")
            IF_MN_GT(0, 11, "ld1 {v19.8h}, [x1], #16\n")

            IF_MN_GT(1, 0, "ld1 {v20.8h}, [x2], #16\n")
            IF_MN_GT(1, 1, "ld1 {v21.8h}, [x2], #16\n")
            IF_MN_GT(1, 2, "ld1 {v22.8h}, [x2], #16\n")
            IF_MN_GT(1, 3, "ld1 {v23.8h}, [x2], #16\n")
            IF_MN_GT(1, 4, "ld1 {v24.8h}, [x2], #16\n")
            IF_MN_GT(1, 5, "ld1 {v25.8h}, [x2], #16\n")
            IF_MN_GT(1, 6, "ld1 {v26.8h}, [x2], #16\n")
            IF_MN_GT(1, 7, "ld1 {v27.8h}, [x2], #16\n")
            IF_MN_GT(1, 8, "ld1 {v28.8h}, [x2], #16\n")
            IF_MN_GT(1, 9, "ld1 {v29.8h}, [x2], #16\n")
            IF_MN_GT(1, 10, "ld1 {v30.8h}, [x2], #16\n")
            IF_MN_GT(1, 11, "ld1 {v31.8h}, [x2], #16\n")

            IF_M_GT(0, "ld1 {v0.8h}, [%[a_ptr]], #16\n")
            IF_N_GT(0, "ld1 {v2.8h}, [%[b_ptr]], #16\n")
            "b 2f\n"

            "1:\n"
            IF_MN_GT(0, 0, "eor v8.16b, v8.16b, v8.16b\n")
            IF_MN_GT(0, 1, "eor v9.16b, v9.16b, v9.16b\n")
            IF_MN_GT(0, 2, "eor v10.16b, v10.16b, v10.16b\n")
            "prfm pstl1keep, [%[outptr0]]\n"
            IF_MN_GT(0, 3, "eor v11.16b, v11.16b, v11.16b\n")
            IF_MN_GT(0, 4, "eor v12.16b, v12.16b, v12.16b\n")
            IF_MN_GT(0, 5, "eor v13.16b, v13.16b, v13.16b\n")
            "prfm pstl1keep, [%[outptr1]]\n"
            IF_MN_GT(0, 6, "eor v14.16b, v14.16b, v14.16b\n")
            IF_MN_GT(0, 7, "eor v15.16b, v15.16b, v15.16b\n")
            IF_MN_GT(0, 8, "eor v16.16b, v16.16b, v16.16b\n")
            IF_N_GT(0, "ld1 {v2.8h}, [%[b_ptr]], #16\n")
            IF_MN_GT(0, 9, "eor v17.16b, v17.16b, v17.16b\n")
            IF_MN_GT(0, 10, "eor v18.16b, v18.16b, v18.16b\n")
            IF_MN_GT(0, 11, "eor v19.16b, v19.16b, v19.16b\n")
            IF_MN_GT(1, 0, "eor v20.16b, v20.16b, v20.16b\n")
            IF_MN_GT(1, 1, "eor v21.16b, v21.16b, v21.16b\n")
            IF_M_GT(0, "ld1 {v0.8h}, [%[a_ptr]], #16\n")
            IF_MN_GT(1, 2, "eor v22.16b, v22.16b, v22.16b\n")
            IF_MN_GT(1, 3, "eor v23.16b, v23.16b, v23.16b\n")
            IF_MN_GT(1, 4, "eor v24.16b, v24.16b, v24.16b\n")
            IF_MN_GT(1, 5, "eor v25.16b, v25.16b, v25.16b\n")
            IF_MN_GT(1, 6, "eor v26.16b, v26.16b, v26.16b\n")
            IF_MN_GT(1, 7, "eor v27.16b, v27.16b, v27.16b\n")
            IF_MN_GT(1, 8, "eor v28.16b, v28.16b, v28.16b\n")
            IF_MN_GT(1, 9, "eor v29.16b, v29.16b, v29.16b\n")
            IF_MN_GT(1, 10, "eor v30.16b, v30.16b, v30.16b\n")
            IF_MN_GT(1, 11, "eor v31.16b, v31.16b, v31.16b\n")

            "2:\n"
            "cmp %w[K], #0\n"
            "beq 4f\n"

            "3:\n"
            "ld1 {v3.8h}, [%[b_ptr]], #16\n"
            IF_MN_GT(0, 0, "fmla v8.8h, v0.8h, v2.h[0]\n")
            IF_MN_GT(0, 1, "fmla v9.8h, v0.8h, v2.h[1]\n")
            IF_MN_GT(0, 2, "fmla v10.8h, v0.8h, v2.h[2]\n")
            IF_MN_GT(0, 3, "fmla v11.8h, v0.8h, v2.h[3]\n")
            IF_M_GT(1, "ld1 {v1.8h}, [%[a_ptr]], #16\n")
            IF_MN_GT(0, 4, "fmla v12.8h, v0.8h, v2.h[4]\n")
            IF_MN_GT(0, 5, "fmla v13.8h, v0.8h, v2.h[5]\n")
            IF_MN_GT(0, 6, "fmla v14.8h, v0.8h, v2.h[6]\n")
            IF_MN_GT(0, 7, "fmla v15.8h, v0.8h, v2.h[7]\n")
            IF_M_GT(0, "ld1 {v5.8h}, [%[a_ptr]], #16\n")
            IF_MN_GT(0, 8, "fmla v16.8h, v0.8h, v3.h[0]\n")
            IF_MN_GT(0, 9, "fmla v17.8h, v0.8h, v3.h[1]\n")
            IF_MN_GT(0, 10, "fmla v18.8h, v0.8h, v3.h[2]\n")
            IF_MN_GT(0, 11, "fmla v19.8h, v0.8h, v3.h[3]\n")

            IF_MN_GT(1, 0, "fmla v20.8h, v1.8h, v2.h[0]\n")
            IF_MN_GT(1, 1, "fmla v21.8h, v1.8h, v2.h[1]\n")
            IF_MN_GT(1, 2, "fmla v22.8h, v1.8h, v2.h[2]\n")
            IF_MN_GT(1, 3, "fmla v23.8h, v1.8h, v2.h[3]\n")
            IF_MN_GT(1, 4, "fmla v24.8h, v1.8h, v2.h[4]\n")
            IF_MN_GT(1, 5, "fmla v25.8h, v1.8h, v2.h[5]\n")
            "ld1 {v4.8h}, [%[b_ptr]], #16\n"
            IF_MN_GT(1, 6, "fmla v26.8h, v1.8h, v2.h[6]\n")
            IF_MN_GT(1, 7, "fmla v27.8h, v1.8h, v2.h[7]\n")
            IF_MN_GT(1, 8, "fmla v28.8h, v1.8h, v3.h[0]\n")
            IF_MN_GT(1, 9, "fmla v29.8h, v1.8h, v3.h[1]\n")
            IF_MN_GT(1, 10, "fmla v30.8h, v1.8h, v3.h[2]\n")
            IF_MN_GT(1, 11, "fmla v31.8h, v1.8h, v3.h[3]\n")

            IF_M_GT(1, "ld1 {v6.8h}, [%[a_ptr]], #16\n")
            IF_MN_GT(0, 0, "fmla v8.8h, v5.8h, v3.h[4]\n")
            IF_MN_GT(0, 1, "fmla v9.8h, v5.8h, v3.h[5]\n")
            IF_M_GT(0, "ld1 {v0.8h}, [%[a_ptr]], #16\n")
            IF_MN_GT(0, 2, "fmla v10.8h, v5.8h, v3.h[6]\n")
            IF_MN_GT(0, 3, "fmla v11.8h, v5.8h, v3.h[7]\n")
            IF_MN_GT(0, 4, "fmla v12.8h, v5.8h, v4.h[0]\n")
            IF_MN_GT(0, 5, "fmla v13.8h, v5.8h, v4.h[1]\n")
            "ld1 {v2.8h}, [%[b_ptr]], #16\n"
            IF_MN_GT(0, 6, "fmla v14.8h, v5.8h, v4.h[2]\n")
            IF_MN_GT(0, 7, "fmla v15.8h, v5.8h, v4.h[3]\n")
            IF_MN_GT(0, 8, "fmla v16.8h, v5.8h, v4.h[4]\n")
            IF_MN_GT(0, 9, "fmla v17.8h, v5.8h, v4.h[5]\n")
            IF_MN_GT(0, 10, "fmla v18.8h, v5.8h, v4.h[6]\n")
            IF_MN_GT(0, 11, "fmla v19.8h, v5.8h, v4.h[7]\n")

            IF_MN_GT(1, 0, "fmla v20.8h, v6.8h, v3.h[4]\n")
            IF_MN_GT(1, 1, "fmla v21.8h, v6.8h, v3.h[5]\n")
            IF_MN_GT(1, 2, "fmla v22.8h, v6.8h, v3.h[6]\n")
            IF_MN_GT(1, 3, "fmla v23.8h, v6.8h, v3.h[7]\n")
            IF_MN_GT(1, 4, "fmla v24.8h, v6.8h, v4.h[0]\n")
            IF_MN_GT(1, 5, "fmla v25.8h, v6.8h, v4.h[1]\n")
            "subs %w[K], %w[K], #1\n"
            IF_MN_GT(1, 6, "fmla v26.8h, v6.8h, v4.h[2]\n")
            IF_MN_GT(1, 7, "fmla v27.8h, v6.8h, v4.h[3]\n")
            IF_MN_GT(1, 8, "fmla v28.8h, v6.8h, v4.h[4]\n")
            IF_MN_GT(1, 9, "fmla v29.8h, v6.8h, v4.h[5]\n")
            IF_MN_GT(1, 10, "fmla v30.8h, v6.8h, v4.h[6]\n")
            IF_MN_GT(1, 11, "fmla v31.8h, v6.8h, v4.h[7]\n")

            "bne 3b\n"

            "4:\n"
            "cmp %w[oddK], #1\n"
            "beq 5f\n"

            // even tail
            "ld1 {v3.8h}, [%[b_ptr]], #16\n"
            IF_MN_GT(0, 0, "fmla v8.8h, v0.8h, v2.h[0]\n")
            IF_MN_GT(0, 1, "fmla v9.8h, v0.8h, v2.h[1]\n")
            IF_MN_GT(0, 2, "fmla v10.8h, v0.8h, v2.h[2]\n")
            IF_MN_GT(0, 3, "fmla v11.8h, v0.8h, v2.h[3]\n")
            IF_M_GT(1, "ld1 {v1.8h}, [%[a_ptr]], #16\n")
            IF_MN_GT(0, 4, "fmla v12.8h, v0.8h, v2.h[4]\n")
            IF_MN_GT(0, 5, "fmla v13.8h, v0.8h, v2.h[5]\n")
            IF_MN_GT(0, 6, "fmla v14.8h, v0.8h, v2.h[6]\n")
            IF_MN_GT(0, 7, "fmla v15.8h, v0.8h, v2.h[7]\n")
            IF_M_GT(0, "ld1 {v5.8h}, [%[a_ptr]], #16\n")
            IF_MN_GT(0, 8, "fmla v16.8h, v0.8h, v3.h[0]\n")
            IF_MN_GT(0, 9, "fmla v17.8h, v0.8h, v3.h[1]\n")
            IF_MN_GT(0, 10, "fmla v18.8h, v0.8h, v3.h[2]\n")
            IF_MN_GT(0, 11, "fmla v19.8h, v0.8h, v3.h[3]\n")

            IF_MN_GT(1, 0, "fmla v20.8h, v1.8h, v2.h[0]\n")
            IF_MN_GT(1, 1, "fmla v21.8h, v1.8h, v2.h[1]\n")
            IF_MN_GT(1, 2, "fmla v22.8h, v1.8h, v2.h[2]\n")
            IF_MN_GT(1, 3, "fmla v23.8h, v1.8h, v2.h[3]\n")
            IF_MN_GT(1, 4, "fmla v24.8h, v1.8h, v2.h[4]\n")
            IF_MN_GT(1, 5, "fmla v25.8h, v1.8h, v2.h[5]\n")
            "ld1 {v4.8h}, [%[b_ptr]], #16\n"
            IF_MN_GT(1, 6, "fmla v26.8h, v1.8h, v2.h[6]\n")
            IF_MN_GT(1, 7, "fmla v27.8h, v1.8h, v2.h[7]\n")
            IF_MN_GT(1, 8, "fmla v28.8h, v1.8h, v3.h[0]\n")
            IF_MN_GT(1, 9, "fmla v29.8h, v1.8h, v3.h[1]\n")
            IF_MN_GT(1, 10, "fmla v30.8h, v1.8h, v3.h[2]\n")
            IF_MN_GT(1, 11, "fmla v31.8h, v1.8h, v3.h[3]\n")

            IF_M_GT(1, "ld1 {v6.8h}, [%[a_ptr]], #16\n")
            IF_MN_GT(0, 0, "fmla v8.8h, v5.8h, v3.h[4]\n")
            IF_MN_GT(0, 1, "fmla v9.8h, v5.8h, v3.h[5]\n")
            IF_MN_GT(0, 2, "fmla v10.8h, v5.8h, v3.h[6]\n")
            IF_MN_GT(0, 3, "fmla v11.8h, v5.8h, v3.h[7]\n")
            IF_MN_GT(0, 4, "fmla v12.8h, v5.8h, v4.h[0]\n")
            IF_MN_GT(0, 5, "fmla v13.8h, v5.8h, v4.h[1]\n")
            IF_MN_GT(0, 6, "fmla v14.8h, v5.8h, v4.h[2]\n")
            IF_MN_GT(0, 7, "fmla v15.8h, v5.8h, v4.h[3]\n")
            IF_MN_GT(0, 0, "st1 {v8.8h}, [%[outptr0]], #16\n")
            IF_MN_GT(0, 8, "fmla v16.8h, v5.8h, v4.h[4]\n")
            IF_MN_GT(0, 1, "st1 {v9.8h}, [%[outptr0]], #16\n")
            IF_MN_GT(0, 9, "fmla v17.8h, v5.8h, v4.h[5]\n")
            IF_MN_GT(0, 2, "st1 {v10.8h}, [%[outptr0]], #16\n")
            IF_MN_GT(0, 10, "fmla v18.8h, v5.8h, v4.h[6]\n")
            IF_MN_GT(0, 3, "st1 {v11.8h}, [%[outptr0]], #16\n")
            IF_MN_GT(0, 11, "fmla v19.8h, v5.8h, v4.h[7]\n")
            IF_MN_GT(0, 4, "st1 {v12.8h}, [%[outptr0]], #16\n")

            IF_MN_GT(1, 0, "fmla v20.8h, v6.8h, v3.h[4]\n")
            IF_MN_GT(0, 5, "st1 {v13.8h}, [%[outptr0]], #16\n")
            IF_MN_GT(1, 1, "fmla v21.8h, v6.8h, v3.h[5]\n")
            IF_MN_GT(0, 6, "st1 {v14.8h}, [%[outptr0]], #16\n")
            IF_MN_GT(1, 2, "fmla v22.8h, v6.8h, v3.h[6]\n")
            IF_MN_GT(0, 7, "st1 {v15.8h}, [%[outptr0]], #16\n")
            IF_MN_GT(1, 3, "fmla v23.8h, v6.8h, v3.h[7]\n")
            IF_MN_GT(0, 8, "st1 {v16.8h}, [%[outptr0]], #16\n")
            IF_MN_GT(1, 4, "fmla v24.8h, v6.8h, v4.h[0]\n")
            IF_MN_GT(0, 9, "st1 {v17.8h}, [%[outptr0]], #16\n")
            IF_MN_GT(1, 5, "fmla v25.8h, v6.8h, v4.h[1]\n")
            IF_MN_GT(0, 10, "st1 {v18.8h}, [%[outptr0]], #16\n")
            IF_MN_GT(1, 6, "fmla v26.8h, v6.8h, v4.h[2]\n")
            IF_MN_GT(0, 11, "st1 {v19.8h}, [%[outptr0]], #16\n")
            IF_MN_GT(1, 7, "fmla v27.8h, v6.8h, v4.h[3]\n")
            IF_MN_GT(1, 0, "st1 {v20.8h}, [%[outptr1]], #16\n")
            IF_MN_GT(1, 8, "fmla v28.8h, v6.8h, v4.h[4]\n")
            IF_MN_GT(1, 1, "st1 {v21.8h}, [%[outptr1]], #16\n")
            IF_MN_GT(1, 9, "fmla v29.8h, v6.8h, v4.h[5]\n")
            IF_MN_GT(1, 2, "st1 {v22.8h}, [%[outptr1]], #16\n")
            IF_MN_GT(1, 10, "fmla v30.8h, v6.8h, v4.h[6]\n")
            IF_MN_GT(1, 3, "st1 {v23.8h}, [%[outptr1]], #16\n")
            IF_MN_GT(1, 11, "fmla v31.8h, v6.8h, v4.h[7]\n")

            IF_MN_GT(1, 4, "st1 {v24.8h}, [%[outptr1]], #16\n")
            IF_MN_GT(1, 5, "st1 {v25.8h}, [%[outptr1]], #16\n")
            IF_MN_GT(1, 6, "st1 {v26.8h}, [%[outptr1]], #16\n")
            IF_MN_GT(1, 7, "st1 {v27.8h}, [%[outptr1]], #16\n")
            IF_MN_GT(1, 8, "st1 {v28.8h}, [%[outptr1]], #16\n")
            IF_MN_GT(1, 9, "st1 {v29.8h}, [%[outptr1]], #16\n")
            IF_MN_GT(1, 10, "st1 {v30.8h}, [%[outptr1]], #16\n")
            IF_MN_GT(1, 11, "st1 {v31.8h}, [%[outptr1]], #16\n")

            "b 6f\n"

            "5:\n"
            // odd tail
            "ld1 {v3.4h}, [%[b_ptr]], #8\n"
            IF_MN_GT(0, 0, "fmla v8.8h, v0.8h, v2.h[0]\n")
            IF_MN_GT(0, 1, "fmla v9.8h, v0.8h, v2.h[1]\n")
            IF_MN_GT(0, 2, "fmla v10.8h, v0.8h, v2.h[2]\n")
            IF_MN_GT(0, 3, "fmla v11.8h, v0.8h, v2.h[3]\n")
            IF_MN_GT(0, 4, "fmla v12.8h, v0.8h, v2.h[4]\n")
            IF_MN_GT(0, 5, "fmla v13.8h, v0.8h, v2.h[5]\n")
            IF_M_GT(1, "ld1 {v1.8h}, [%[a_ptr]], #16\n")
            IF_MN_GT(0, 6, "fmla v14.8h, v0.8h, v2.h[6]\n")
            IF_MN_GT(0, 7, "fmla v15.8h, v0.8h, v2.h[7]\n")
            IF_MN_GT(0, 0, "st1 {v8.8h}, [%[outptr0]], #16\n")
            IF_MN_GT(0, 8, "fmla v16.8h, v0.8h, v3.h[0]\n")
            IF_MN_GT(0, 1, "st1 {v9.8h}, [%[outptr0]], #16\n")
            IF_MN_GT(0, 9, "fmla v17.8h, v0.8h, v3.h[1]\n")
            IF_MN_GT(0, 2, "st1 {v10.8h}, [%[outptr0]], #16\n")
            IF_MN_GT(0, 10, "fmla v18.8h, v0.8h, v3.h[2]\n")
            IF_MN_GT(0, 3, "st1 {v11.8h}, [%[outptr0]], #16\n")
            IF_MN_GT(0, 11, "fmla v19.8h, v0.8h, v3.h[3]\n")
            IF_MN_GT(0, 4, "st1 {v12.8h}, [%[outptr0]], #16\n")

            IF_MN_GT(1, 0, "fmla v20.8h, v1.8h, v2.h[0]\n")
            IF_MN_GT(0, 5, "st1 {v13.8h}, [%[outptr0]], #16\n")
            IF_MN_GT(1, 1, "fmla v21.8h, v1.8h, v2.h[1]\n")
            IF_MN_GT(0, 6, "st1 {v14.8h}, [%[outptr0]], #16\n")
            IF_MN_GT(1, 2, "fmla v22.8h, v1.8h, v2.h[2]\n")
            IF_MN_GT(0, 7, "st1 {v15.8h}, [%[outptr0]], #16\n")
            IF_MN_GT(1, 3, "fmla v23.8h, v1.8h, v2.h[3]\n")
            IF_MN_GT(0, 8, "st1 {v16.8h}, [%[outptr0]], #16\n")
            IF_MN_GT(1, 4, "fmla v24.8h, v1.8h, v2.h[4]\n")
            IF_MN_GT(0, 9, "st1 {v17.8h}, [%[outptr0]], #16\n")
            IF_MN_GT(1, 5, "fmla v25.8h, v1.8h, v2.h[5]\n")
            IF_MN_GT(0, 10, "st1 {v18.8h}, [%[outptr0]], #16\n")
            IF_MN_GT(1, 6, "fmla v26.8h, v1.8h, v2.h[6]\n")
            IF_MN_GT(0, 11, "st1 {v19.8h}, [%[outptr0]], #16\n")
            IF_MN_GT(1, 7, "fmla v27.8h, v1.8h, v2.h[7]\n")
            IF_MN_GT(1, 0, "st1 {v20.8h}, [%[outptr1]], #16\n")
            IF_MN_GT(1, 8, "fmla v28.8h, v1.8h, v3.h[0]\n")
            IF_MN_GT(1, 1, "st1 {v21.8h}, [%[outptr1]], #16\n")
            IF_MN_GT(1, 9, "fmla v29.8h, v1.8h, v3.h[1]\n")
            IF_MN_GT(1, 2, "st1 {v22.8h}, [%[outptr1]], #16\n")
            IF_MN_GT(1, 10, "fmla v30.8h, v1.8h, v3.h[2]\n")
            IF_MN_GT(1, 3, "st1 {v23.8h}, [%[outptr1]], #16\n")
            IF_MN_GT(1, 11, "fmla v31.8h, v1.8h, v3.h[3]\n")

            IF_MN_GT(1, 4, "st1 {v24.8h}, [%[outptr1]], #16\n")
            IF_MN_GT(1, 5, "st1 {v25.8h}, [%[outptr1]], #16\n")
            IF_MN_GT(1, 6, "st1 {v26.8h}, [%[outptr1]], #16\n")
            IF_MN_GT(1, 7, "st1 {v27.8h}, [%[outptr1]], #16\n")
            IF_MN_GT(1, 8, "st1 {v28.8h}, [%[outptr1]], #16\n")
            IF_MN_GT(1, 9, "st1 {v29.8h}, [%[outptr1]], #16\n")
            IF_MN_GT(1, 10, "st1 {v30.8h}, [%[outptr1]], #16\n")
            IF_MN_GT(1, 11, "st1 {v31.8h}, [%[outptr1]], #16\n")

            "6:\n"
            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
                [is_first_k] "+r"(is_first_k), [oddK] "+r"(oddK),
                [outptr0] "+r"(outptr0), [outptr1] "+r"(outptr1)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v8", "v9", "v10", "v11", "v12", "v13",
                "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
                "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x1", "x2",
                "cc", "memory");
#undef IF_MN_GT
#undef IF_N_GT
#undef IF_M_GT
}
#endif