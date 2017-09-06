/* Author: Changho Hwang         */
/* E-mail: ch.hwang128@gmail.com */

/* reduction/main.cu */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <assert.h>
#include "../common/cputimer.h"
#include "reduction.cuh"

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

/* usage */
static void
print_usage(const char *prog)
{
    printf("Usage: %s NUM_INTEGERS NUM_STREAMS\n" \
           "  NUM_INTEGERS - # of integers to copy at once\n" \
           "  NUM_STREAMS  - # of CUDA streams to use\n",
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
    const int nint = atoi(argv[1]);
    const unsigned int memsize = nint * sizeof(int);
    const unsigned int nloop = 100;
    const unsigned int block_size = 1024;
    int *h_data, *d_data, *d_res;
    int data;
    double start, elapsed;
    unsigned int i;

    printf("Reduction Kernel Test: %d Integers\n", nint);
    assert(nint > 0);
    assert((nint % 2) == 0);

    // allocate memory
    h_data = (int *)malloc(memsize);
    if (h_data == NULL) {
        perror("malloc");
        return -1;
    }
    cudaErrCheck( cudaMalloc((void **)&d_data, memsize) );
    cudaErrCheck( cudaMalloc((void **)&d_res, memsize/block_size + 1) );

    // fill in the data
    for (data = 0; data < nint/2; data++) {
        h_data[2*data] = data;
        h_data[2*data+1] = -data;
    }
    assert(data != 0);

    // copy the data to GPU
    cudaErrCheck( cudaMemcpy(d_data, h_data, memsize, cudaMemcpyHostToDevice) );

    // test correctness of the kernel
    reduction<int>((const int *)d_data, d_res, nint, block_size);
    cudaErrCheck( cudaGetLastError() );
    reduction<int>((const int *)d_data, d_res, nint, block_size);
    cudaErrCheck( cudaGetLastError() );
    cudaErrCheck( cudaMemcpy(&data, d_res, sizeof(int), cudaMemcpyDeviceToHost) );
    if (data != 0) {
        printf("  Correctness Test Failed!\n");
        return -1;
    }
    printf("  Passed Correctness Test\n");

    // performance test
    start = cputimer_sec();
    for (i = 0; i < nloop; i++) {
        reduction<int>((const int *)d_data, d_res, nint, block_size);
    }
    cudaErrCheck( cudaDeviceSynchronize() );
    elapsed = cputimer_sec() - start;
    printf("  Elapsed: %.1f msec, Bandwidth: %f GB/s\n",
           elapsed * 1000, memsize / 1073741824.0 * nloop / elapsed);

    cudaErrCheck( cudaFree(d_data) );
    cudaErrCheck( cudaFree(d_res) );
    free(h_data);
    return 0;
}
