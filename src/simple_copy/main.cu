/* Author: Changho Hwang         */
/* E-mail: ch.hwang128@gmail.com */

/* simple_copy/main.cu */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include "../common/cputimer.h"

/* CUDA error checker */
#define cudaErrCheck(res)   { cudaErrCheck_helper((res), __FILE__, __LINE__); }
static cudaError_t
cudaErrCheck_helper(cudaError_t result, const char *file, int line)
{
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s at %s, line %d\n",
                cudaGetErrorString(result), file, line);
        exit(result);
    }
    return result;
}

/* Null kernel */
__global__ void
kernel_null(void)
{
    return; // do nothing, just return
}

/* usage */
static void
print_usage(const char *prog)
{
    printf("Usage: %s NUM_INTEGERS\n" \
           "  NUM_INTEGERS - # of integers to copy at once\n",
           prog);
    return;
}

int
main(int argc, const char **argv)
{
    if (argc != 2) {
        print_usage(argv[0]);
        return -1;
    }
    const unsigned int memsize = atoi(argv[1]) * sizeof(int);
    const unsigned int nloop = 100;
    char *h_data_pagable, *h_data_pinned;
    char *d_data;
    double start, elapsed;
    unsigned int i;

    printf("Memory Copy Test: %f GB\n", memsize / 1073741824.0);
    
    h_data_pagable = (char *)malloc(memsize);
    if (h_data_pagable == NULL) {
        perror("malloc");
        return -1;
    }
    cudaErrCheck( cudaMallocHost((void **)&h_data_pinned, memsize) );
    cudaErrCheck( cudaMalloc((void **)&d_data, memsize) );

    // test pagable memory copy
    printf("Pagable Memory Copy\n");
    start = cputimer_sec();
    for (i = 0; i < nloop; i++) {
        cudaErrCheck( cudaMemcpy(d_data, h_data_pagable, memsize,
                                 cudaMemcpyHostToDevice) );
        kernel_null<<<1, 1>>>();
        cudaErrCheck( cudaGetLastError() );
        cudaErrCheck( cudaMemcpy(h_data_pagable, d_data, memsize,
                                 cudaMemcpyDeviceToHost) );
    }
    elapsed = cputimer_sec() - start;
    printf("  Elapsed: %.1f msec, Bandwidth: %.2f GB/s\n",
           elapsed * 1000, memsize / 1073741824.0 * nloop / elapsed);

    // test pinned memory copy
    printf("Pinned Memory Copy\n");
    start = cputimer_sec();
    for (i = 0; i < nloop; i++) {
        cudaErrCheck( cudaMemcpy(d_data, h_data_pinned, memsize,
                                 cudaMemcpyHostToDevice) );
        kernel_null<<<1, 1>>>();
        cudaErrCheck( cudaGetLastError() );
        cudaErrCheck( cudaMemcpy(h_data_pinned, d_data, memsize,
                                 cudaMemcpyDeviceToHost) );
    }
    elapsed = cputimer_sec() - start;

    printf("  Elapsed: %.1f msec, Bandwidth: %.2f GB/s\n",
           elapsed * 1000, memsize / 1073741824.0 * nloop / elapsed);

    free(h_data_pagable);
    cudaErrCheck( cudaFreeHost(h_data_pinned) );
    cudaErrCheck( cudaFree(d_data) );
    return 0;
}
