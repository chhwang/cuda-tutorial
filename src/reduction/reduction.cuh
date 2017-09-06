/* Author: Changho Hwang         */
/* E-mail: ch.hwang128@gmail.com */

/* reduction/reduction.cuh */

#ifndef REDUCTION_CUH
#define REDUCTION_CUH

/*----------------------------------------------------------------------------*/
/* Utility class used to avoid linker errors with extern unsized shared
   memory arrays with templated type */
/*----------------------------------------------------------------------------*/
template <typename T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};
/*----------------------------------------------------------------------------*/
/* Specialize for double to avoid unaligned memory access compile errors */
/*----------------------------------------------------------------------------*/
template <>
struct SharedMemory <double>
{
    __device__ inline operator       double *()
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }
    __device__ inline operator const double *() const
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }
};

/*----------------------------------------------------------------------------*/
/* Partial reduction GPU kernel.
   Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
   In other words if block_size <= 32, allocate 64*sizeof(T) bytes.
   If block_size > 32, allocate block_size*sizeof(T) bytes. */
/*----------------------------------------------------------------------------*/
template <typename T, unsigned int block_size, bool ispow2>
__device__ T
partial_sum_kernel(const T *in_buf, const unsigned int size)
{
    T *sdata = SharedMemory<T>();
    T sum = 0;

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * block_size * 2 + threadIdx.x;
    unsigned int grid_size = block_size * 2 * gridDim.x;

    /* The first step - read from global mem, write on shared mem */
    while (i < size) {
        sum += in_buf[i];

        /* ensure we don't read out of bounds.
           this is optimized away for power-of-2 sized arrays. */
        if (ispow2 || i + block_size < size)
            sum += in_buf[i + block_size];

        i += grid_size;
    }

    /* each thread puts its local sum into shared memory */
    sdata[tid] = sum;
    __syncthreads();


    /* The second step - do reduction in shared mem */
    if ((block_size >= 512) && (tid < 256))
        sdata[tid] = sum = sum + sdata[tid + 256];

    __syncthreads();

    if ((block_size >= 256) && (tid < 128))
        sdata[tid] = sum = sum + sdata[tid + 128];

     __syncthreads();

    if ((block_size >= 128) && (tid < 64))
       sdata[tid] = sum = sum + sdata[tid + 64];

    __syncthreads();

#if (__CUDA_ARCH__ >= 300)
    if (tid < 32) {
        /* Fetch final intermediate sum from 2nd warp */
        if (block_size >= 64)
            sum += sdata[tid + 32];
        /* Reduce final warp using shuffle */
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
            sum += __shfl_down(sum, offset);
    }
#else
    /* fully unroll reduction within a single warp */
    if ((block_size >= 64) && (tid < 32))
        sdata[tid] = sum = sum + sdata[tid + 32];

    __syncthreads();

    if ((block_size >= 32) && (tid < 16))
        sdata[tid] = sum = sum + sdata[tid + 16];

    __syncthreads();

    if ((block_size >= 16) && (tid < 8))
        sdata[tid] = sum = sum + sdata[tid + 8];

    __syncthreads();

    if ((block_size >= 8) && (tid < 4))
        sdata[tid] = sum = sum + sdata[tid + 4];

    __syncthreads();

    if ((block_size >= 4) && (tid < 2))
        sdata[tid] = sum = sum + sdata[tid + 2];

    __syncthreads();

    if ((block_size >= 2) && (tid < 1))
        sdata[tid] = sum = sum + sdata[tid + 1];

    __syncthreads();
#endif
    return sum;
}

/*----------------------------------------------------------------------------*/
/* Reduction kernel. */
/*----------------------------------------------------------------------------*/
__device__ unsigned int num_done = 0;
template <typename T, unsigned int block_size, bool ispow2>
__global__ void
reduction_kernel(const T *in_buf, volatile T *out_buf,
                 const unsigned int size)
{
    __shared__ bool all_done;
    all_done = false;
    T sum = partial_sum_kernel<T, block_size, ispow2>(in_buf, size);
    if (threadIdx.x == 0) {
        /* Thread 0 of each block stores the partial sum to global memory.
           The compiler will use a store operation that bypasses the L1 cache
           since out_buf is declared as volatile. This ensures that the threads
           of the last block will read the correct partial sums by all other
           blocks. The memory fence ensures that thread 0 proceeds to the next
           operation after the partial sum has been written to global memory. */
        out_buf[blockIdx.x] = sum;
        __threadfence();

        /* Each block globally informs that itself is done, and the last block
           informs to threads in that block that every blocks are done. */
        all_done = (atomicInc(&num_done, gridDim.x) == (gridDim.x - 1));
    }

    __syncthreads();

    if (all_done) {
        /* Only the last block reaches here */
        sum = partial_sum_kernel<T, block_size, ispow2>(in_buf, gridDim.x);
        if (threadIdx.x == 0) {
            /* write the total sum */
            out_buf[0] = sum;

            /* reset num_done for the next execution of this kernel */
            num_done = 0;
        }
    }
}
/*----------------------------------------------------------------------------*/
/* GPU parallel reduction wrapper function
     in_buf     - device pointer to the input array
     out_buf    - device pointer to the output array
     size       - length of the input array
     block_size - # of threads per block */
/* Prerequisite 1: block_size should be power of 2 */
/* Prerequisite 2: out_buf should be allocated at least
                   ceil(size/block_size)*sizeof(T) bytes of device memory */
/* Deciding block_size is a design choice, but typically use
   256, 512, or 1024 for large amount of data to process */
/*----------------------------------------------------------------------------*/
template <typename T>
void
reduction(const T *in_buf, T *out_buf,
          const unsigned int size, const unsigned int block_size)
{
    /* when there is only one warp per block, we need to allocate two warps
       worth of shared memory so that we don't index shared memory
       out of bounds */
    unsigned int smem_size = (block_size <= 32) ? 2 * block_size * sizeof(T) + 1
                                                : block_size * sizeof(T) + 1;
    /* nblock = ceil(size/block_size) */
    unsigned int nblock = size / block_size;
    if ((size & (block_size - 1)) != 0) {
        nblock++;
    }
    if ((size & (size - 1)) == 0) {
        switch (block_size) {
        case 1024:
            reduction_kernel<T, 1024, true><<<nblock, block_size, smem_size>>>
                                           (in_buf, out_buf, size);
            break;
        case 512:
            reduction_kernel<T,  512, true><<<nblock, block_size, smem_size>>>
                                           (in_buf, out_buf, size);
            break;
        case 256:
            reduction_kernel<T,  256, true><<<nblock, block_size, smem_size>>>
                                           (in_buf, out_buf, size);
            break;
        case 128:
            reduction_kernel<T,  128, true><<<nblock, block_size, smem_size>>>
                                           (in_buf, out_buf, size);
            break;
        case 64:
            reduction_kernel<T,   64, true><<<nblock, block_size, smem_size>>>
                                           (in_buf, out_buf, size);
            break;
        case 32:
            reduction_kernel<T,   32, true><<<nblock, block_size, smem_size>>>
                                           (in_buf, out_buf, size);
            break;
        case 16:
            reduction_kernel<T,   16, true><<<nblock, block_size, smem_size>>>
                                           (in_buf, out_buf, size);
            break;
        case  8:
            reduction_kernel<T,    8, true><<<nblock, block_size, smem_size>>>
                                           (in_buf, out_buf, size);
            break;
        case  4:
            reduction_kernel<T,    4, true><<<nblock, block_size, smem_size>>>
                                           (in_buf, out_buf, size);
            break;
        case  2:
            reduction_kernel<T,    2, true><<<nblock, block_size, smem_size>>>
                                           (in_buf, out_buf, size);
            break;
        case  1:
            reduction_kernel<T,    1, true><<<nblock, block_size, smem_size>>>
                                           (in_buf, out_buf, size);
            break;
        }
    } else {
        switch (block_size) {
        case 1024:
            reduction_kernel<T, 1024, false><<<nblock, block_size, smem_size>>>
                                            (in_buf, out_buf, size);
            break;
        case 512:
            reduction_kernel<T,  512, false><<<nblock, block_size, smem_size>>>
                                            (in_buf, out_buf, size);
            break;
        case 256:
            reduction_kernel<T,  256, false><<<nblock, block_size, smem_size>>>
                                            (in_buf, out_buf, size);
            break;
        case 128:
            reduction_kernel<T,  128, false><<<nblock, block_size, smem_size>>>
                                            (in_buf, out_buf, size);
            break;
        case 64:
            reduction_kernel<T,   64, false><<<nblock, block_size, smem_size>>>
                                            (in_buf, out_buf, size);
            break;
        case 32:
            reduction_kernel<T,   32, false><<<nblock, block_size, smem_size>>>
                                            (in_buf, out_buf, size);
            break;
        case 16:
            reduction_kernel<T,   16, false><<<nblock, block_size, smem_size>>>
                                            (in_buf, out_buf, size);
            break;
        case  8:
            reduction_kernel<T,    8, false><<<nblock, block_size, smem_size>>>
                                            (in_buf, out_buf, size);
            break;
        case  4:
            reduction_kernel<T,    4, false><<<nblock, block_size, smem_size>>>
                                            (in_buf, out_buf, size);
            break;
        case  2:
            reduction_kernel<T,    2, false><<<nblock, block_size, smem_size>>>
                                            (in_buf, out_buf, size);
            break;
        case  1:
            reduction_kernel<T,    1, false><<<nblock, block_size, smem_size>>>
                                            (in_buf, out_buf, size);
            break;
        }
    }
}

template void
reduction<int>(const int *, int *,
               const unsigned int, const unsigned int);
template void
reduction<float>(const float *, float *,
                 const unsigned int, const unsigned int);
template void
reduction<double>(const double *, double *,
                  const unsigned int, const unsigned int);

#endif  /* end of REDUCTION_CUH */
