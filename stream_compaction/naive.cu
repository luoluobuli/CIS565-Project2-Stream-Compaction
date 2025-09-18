#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__


        __global__ void kernNaiveScan(int n, int* odata, const int* idata, int* blockSum) {
            int id = blockIdx.x * blockDim.x + threadIdx.x;
            int thid = threadIdx.x;
            int size = blockDim.x;

            if (id >= n) return;

            // Shared memory array
            extern __shared__ int temp[];

            // Set ping pong buffer flags
            int pout = 0, pin = 1;

            // Load array
            temp[pout * size + thid] = (thid > 0) ? idata[id - 1] : 0;  // thread 0 -> 0, others -> idata[id - 1]
            __syncthreads();

            // Scan within the block
            for (int offset = 1; offset < n; offset *= 2) {
                pout = 1 - pout;
                pin = 1 - pout;
                if (thid >= offset) {
                    temp[pout * size + thid] = temp[pin * size + thid] + temp[pin * size + thid - offset];
                }
                else {
                    temp[pout * size + thid] = temp[pin * size + thid];
                }
                __syncthreads();
            }

            odata[id] = temp[pout * size + thid];

            // Write block sum
            if (threadIdx.x == blockDim.x - 1 || id == n - 1) {
                int blockEnd = (n < (blockIdx.x + 1) * size) ? n : ((blockIdx.x + 1) * size); // last global index in this block
                blockSum[blockIdx.x] = odata[blockEnd - 1] + idata[blockEnd - 1];
            }
        }

        __global__ void kernAddBlockOffset(int n, int* odata, const int* blockOffset) {
            int id = blockIdx.x * blockDim.x + threadIdx.x;
            if (id >= n) return;
            odata[id] += blockOffset[blockIdx.x];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            int threads = 1024;
            int blocks = (n + threads - 1) / threads; // ceil
            
            // Allocate & copy memory
            int* d_odata, *d_idata;
            
            cudaMalloc(&d_odata, n * sizeof(int));
            checkCUDAError("Naive::cudaMalloc d_odata fails!");

            cudaMalloc(&d_idata, n * sizeof(int));
            checkCUDAError("Naive::cudaMalloc d_idata fails!");
            
            cudaMemcpy(d_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("Naive::cudaMemcpyHostToDevice fails!");

            // Handle array of arbitrary length - create and set blockSum
            int* blockSum = new int[blocks];

            int* d_blockSum;
            cudaMalloc(&d_blockSum, blocks * sizeof(int));
            checkCUDAError("Naive::cudaMalloc d_blockSum fails!");

            cudaMemset(d_blockSum, 0, blocks * sizeof(int)); // Initialize the sum to 0
            checkCUDAError("Naive::cudaMemset d_blockSum fails!");

            // Call kernNaiveScan
            kernNaiveScan<<<blocks, threads, 2 * threads * sizeof(int)>>>(n, d_odata, d_idata, d_blockSum);
            checkCUDAError("Naive::kernNaiveScan fails!");

            // Handle array of arbitrary length - scan the blockSum (on CPU)
            cudaMemcpy(blockSum, d_blockSum, blocks * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("Naive::cudaMemcpyDeviceToHost fails!");

            int* blockOffset = new int[blocks];
            blockOffset[0] = 0;
            for (int i = 1; i < blocks; ++i) {
                blockOffset[i] = blockOffset[i - 1] + blockSum[i - 1];
            }

            // Handle array of arbitrary length - add the offset back to array
            int* d_blockOffset;
            cudaMalloc(&d_blockOffset, blocks * sizeof(int));
            checkCUDAError("Naive::cudaMalloc d_blockOffset fails!");

            cudaMemcpy(d_blockOffset, blockOffset, blocks * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("Naive::cudaMemcpyHostToDevice fails!");

            // Call kernAddBlockOffset
            kernAddBlockOffset<<<blocks, threads>>>(n, d_odata, d_blockOffset);
            checkCUDAError("Naive::kernAddBlockOffset fails!");

            // Copy the value back
            cudaMemcpy(odata, d_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("Naive::cudaMemcpyDeviceToHost fails!");

            timer().endGpuTimer();

            delete[] blockSum;
            delete[] blockOffset;
            cudaFree(d_idata);
            cudaFree(d_odata);
            cudaFree(d_blockSum);
            cudaFree(d_blockOffset);
        }
    }
}
