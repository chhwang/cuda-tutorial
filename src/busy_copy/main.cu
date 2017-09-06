/* Author: Changho Hwang         */
/* E-mail: ch.hwang128@gmail.com */

/* busy_copy/main.cu */

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
    printf("Usage: %s NUM_INTEGERS NUM_STREAMS\n" \
           "  NUM_INTEGERS - # of integers to copy at once\n" \
           "  NUM_STREAMS  - # of CUDA streams to use\n",
           prog);
    return;
}

int
main(int argc, const char **argv)
{
    if (argc != 3) {
        print_usage(argv[0]);
        return -1;
    }
    const unsigned int memsize = atoi(argv[1]) * sizeof(int);
    const unsigned int nstream = atoi(argv[2]);
    const unsigned int nloop = 100;
    const unsigned int memsize_per_stream = memsize / nstream;
    cudaStream_t *stream;
    char *h_data, *d_data;
    double start, elapsed;
    unsigned int i, sidx;

    printf("Busy DMA Test: %f GB, %u streams\n",
           memsize / 1073741824.0, nstream);

    // allocate memory
    cudaErrCheck( cudaMallocHost((void **)&h_data, memsize) );
    cudaErrCheck( cudaMalloc((void **)&d_data, memsize) );

    // create streams
    stream = (cudaStream_t *)malloc(nstream * sizeof(cudaStream_t));
    if (stream == NULL) {
        perror("malloc");
        return -1;
    }
    for (sidx = 0; sidx < nstream; sidx++) {
        cudaErrCheck( cudaStreamCreate(&stream[sidx]) );
    }

    start = cputimer_sec();
    for (i = 0; i < nloop; i++) {
        for (sidx = 0; sidx < nstream; sidx++) {
            cudaErrCheck( cudaMemcpyAsync(d_data + sidx * memsize_per_stream,
                                          h_data + sidx * memsize_per_stream,
                                          memsize_per_stream,
                                          cudaMemcpyHostToDevice,
                                          stream[sidx]) );
        }
        for (sidx = 0; sidx < nstream; sidx++) {
            kernel_null<<<1, 1, 0, stream[sidx]>>>();
        }
        cudaErrCheck( cudaGetLastError() );
        for (sidx = 0; sidx < nstream; sidx++) {
            cudaErrCheck( cudaMemcpyAsync(h_data + sidx * memsize_per_stream,
                                          d_data + sidx * memsize_per_stream,
                                          memsize_per_stream,
                                          cudaMemcpyDeviceToHost,
                                          stream[sidx]) );
        }
        cudaErrCheck( cudaDeviceSynchronize() );
    }
    elapsed = cputimer_sec() - start;
    printf("  Elapsed: %.1f msec, Bandwidth: %f GB/s\n",
           elapsed * 1000, memsize / 1073741824.0 * nloop / elapsed);

    cudaErrCheck( cudaFreeHost(h_data) );
    cudaErrCheck( cudaFree(d_data) );
    for (sidx = 0; sidx < nstream; sidx++) {
        cudaErrCheck( cudaStreamDestroy(stream[sidx]) );
    }
    free(stream);
    return 0;
}
