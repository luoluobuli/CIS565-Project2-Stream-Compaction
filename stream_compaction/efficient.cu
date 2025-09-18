#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernEfficientScan(int n, int* odata, const int* idata, int* blockSum) {
            int id = blockIdx.x * blockDim.x + threadIdx.x;
            int thid = threadIdx.x;
            int size = 2 * blockDim.x;
            int offset = 1;

            // Shared memory array
            extern __shared__ int temp[];

            // Load input into shared memory
            temp[2 * thid] = idata[2 * id];
            temp[2 * thid + 1] = idata[2 * id + 1];
            __syncthreads();

            // Up sweep
            for (int d = size >> 1; d > 0; d >>= 1) {
                __syncthreads();
                if (thid < d) {
                    int ai = offset * (2 * thid + 1) - 1;
                    int bi = offset * (2 * thid + 2) - 1;
                    temp[bi] += temp[ai];
                }
                offset *= 2;
            }

            // Set the root for down sweep tree to 0
            if (thid == 0) {
                blockSum[blockIdx.x] = temp[size - 1];
                temp[size - 1] = 0;
            }

            // Down sweep
            for (int d = 1; d < size; d *= 2) {
                offset >>= 1;
                __syncthreads();
                if (thid < d) {
                    int ai = offset * (2 * thid + 1) - 1;
                    int bi = offset * (2 * thid + 2) - 1;
                    int t = temp[ai];
                    temp[ai] = temp[bi];
                    temp[bi] += t;
                }
            }
            __syncthreads();

            odata[2 * id] = temp[2 * thid]; // write results to device memory
            odata[2 * id + 1] = temp[2 * thid + 1];
        }


        __global__ void kernAddBlockOffset(int n, int* odata, const int* blockOffset) {
            int id = blockIdx.x * blockDim.x + threadIdx.x;
            if (id >= n / 2) return;
            odata[2 * id] += blockOffset[blockIdx.x];
            odata[2 * id + 1] += blockOffset[blockIdx.x];
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            //timer().startGpuTimer();
            // TODO
            // Pad
            int pad_n = 1 << ilog2ceil(n);

            // Assign hreads and blocks
            int threads = 1024;
            int blocks = (pad_n / 2 + threads - 1) / threads; // ceil

            // Allocate & copy memory
            int* d_odata, *d_idata;
            
            cudaMalloc(&d_odata, pad_n * sizeof(int));
            checkCUDAError("Efficient::scan::cudaMalloc d_odata fails!");

            cudaMalloc(&d_idata, pad_n * sizeof(int));
            checkCUDAError("Efficient::scan::cudaMalloc d_idata fails!");

            cudaMemset(d_idata, 0, pad_n * sizeof(int));
            checkCUDAError("Efficient::scan::cudaMemset d_idata fails!");
            
            cudaMemcpy(d_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("Efficient::scan::cudaMemcpyHostToDevice fails!");

            // Handle array of arbitrary length - create and set blockSum
            int* blockSum = new int[blocks];

            int* d_blockSum;
            cudaMalloc(&d_blockSum, blocks * sizeof(int));
            checkCUDAError("Efficient::scan::cudaMalloc d_blockSum fails!");

            cudaMemset(d_blockSum, 0, blocks * sizeof(int)); // Initialize the sum to 0
            checkCUDAError("Efficient::scan::cudaMemset d_blockSum fails!");

            // Call kernEfficientScan
            kernEfficientScan<<<blocks, threads, 2 * threads * sizeof(int)>>>(pad_n, d_odata, d_idata, d_blockSum);
            checkCUDAError("Efficient::scan::kernEfficientScan fails!");

            // Handle array of arbitrary length - scan the blockSum (on CPU)
            cudaMemcpy(blockSum, d_blockSum, blocks * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("Efficient::scan::cudaMemcpyDeviceToHost fails!");

            int* blockOffset = new int[blocks];
            blockOffset[0] = 0;
            for (int i = 1; i < blocks; ++i) {
                blockOffset[i] = blockOffset[i - 1] + blockSum[i - 1];
            }

            // Handle array of arbitrary length - add the offset back to array
            int* d_blockOffset;
            cudaMalloc(&d_blockOffset, blocks * sizeof(int));
            checkCUDAError("Efficient::scan::cudaMalloc d_blockOffset fails!");

            cudaMemcpy(d_blockOffset, blockOffset, blocks * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("Efficient::scan::cudaMemcpyHostToDevice fails!");

            // Call kernAddBlockOffset
            kernAddBlockOffset<<<blocks, threads>>>(pad_n, d_odata, d_blockOffset);
            checkCUDAError("Efficient::scan::kernAddBlockOffset fails!");

            // Copy the value back
            cudaMemcpy(odata, d_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("Efficient::scan::cudaMemcpyDeviceToHost fails!");

            //timer().endGpuTimer();

            delete[] blockSum;
            delete[] blockOffset;
            cudaFree(d_idata);
            cudaFree(d_odata);
            cudaFree(d_blockSum);
            cudaFree(d_blockOffset);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            int threads = 1024;
            int blocks = (n + threads - 1) / threads; // ceil

            // Allocate memory on device
            int* d_odata, *d_idata, *d_bools, *d_indices;

            cudaMalloc(&d_odata, n * sizeof(int));
            checkCUDAError("Efficient::compact::cudaMalloc d_odata fails!");

            cudaMalloc(&d_idata, n * sizeof(int));
            checkCUDAError("Efficient::compact::cudaMalloc d_idata fails!");

            cudaMalloc(&d_bools, n * sizeof(int));
            checkCUDAError("Efficient::compact::cudaMalloc d_bools fails!");

            cudaMalloc(&d_indices, n * sizeof(int));
            checkCUDAError("Efficient::compact::cudaMalloc d_indices fails!");

            cudaMemcpy(d_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("Efficient::compact::cudaMemcpyHostToDevice fails!");

            // Create boolean array
            Common::kernMapToBoolean<<<blocks, threads>>>(n, d_bools, d_idata);
            checkCUDAError("Efficient::compact::kernMapToBoolean fails!");

            // Create indices array through exclusive scan
            scan(n, d_indices, d_bools);

            // Scatter
            Common:: kernScatter<<<blocks, threads>>>(n, d_odata, d_idata, d_bools, d_indices);
            checkCUDAError("Efficient::compact::kernScatter fails!");

            cudaMemcpy(odata, d_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("Efficient::compact::cudaMemcpyDeviceToHost fails!");

            // Get the count
            int lastIndex, lastBool;
            cudaMemcpy(&lastIndex, d_indices + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastBool, d_bools + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            int count = lastIndex + lastBool;

            timer().endGpuTimer();

            cudaFree(d_odata);
            cudaFree(d_idata);
            cudaFree(d_bools);
            cudaFree(d_indices);

            return count;
        }
    }
}
