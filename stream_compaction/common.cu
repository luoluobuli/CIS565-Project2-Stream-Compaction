#include "common.h"

void checkCUDAErrorFn(const char *msg, const char *file, int line) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}


namespace StreamCompaction {
    namespace Common {

        /**
         * Maps an array to an array of 0s and 1s for stream compaction. Elements
         * which map to 0 will be removed, and elements which map to 1 will be kept.
         */
        __global__ void kernMapToBoolean(int n, int *bools, const int *idata) {
            // TODO
            int id = threadIdx.x + blockIdx.x * blockDim.x;
            if (id >= n) return;

            if (idata[id] == 0) {
                bools[id] = 0;
            } else {
                bools[id] = 1;
            }
        }

        /**
         * Performs scatter on an array. That is, for each element in idata,
         * if bools[idx] == 1, it copies idata[idx] to odata[indices[idx]].
         */
        __global__ void kernScatter(int n, int *odata,
                const int *idata, const int *bools, const int *indices) {
            // TODO
            int id = threadIdx.x + blockIdx.x * blockDim.x;
            if (id >= n) return;

            if (bools[id] == 1) {
                odata[indices[id]] = idata[id];
            }
        }

        __global__ void kernMapToBit(int n, int* revBits, const int* idata, int pass) {
            int id = threadIdx.x + blockIdx.x * blockDim.x;
            if (id >= n) return;

            int bit = (idata[id] >> pass) & 1;
            revBits[id] = 1 - bit;
        }

        __global__ void kernRadixScatter(int n, int totalFalses, int* odata, const int* revBits, const int* falses, const int* idata) {
            int id = threadIdx.x + blockIdx.x * blockDim.x;
            if (id >= n) return;

            int t = id - falses[id] + totalFalses;

            __syncthreads();

            int index = revBits[id] ? falses[id] : t;

            __syncthreads();

            odata[index] = idata[id];
        }

    }
}
