#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/remove.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            thrust::host_vector<int> in(idata, idata + n);
            thrust::device_vector<int> d_in = in;
            thrust::device_vector<int> d_out(n);

            timer().startGpuTimer();

            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
            
            thrust::exclusive_scan(d_in.begin(), d_in.end(), d_out.begin());

            timer().endGpuTimer();

            thrust::copy(d_out.begin(), d_out.end(), odata);
        }

        struct is_zero
        {
            __host__ __device__
                bool operator()(const int x) const {
                return x == 0;
            }
        };

        int compact(int n, int* odata, const int* idata) {
            thrust::device_vector<int> d_in(idata, idata + n);

            timer().startGpuTimer();

            auto new_end = thrust::remove_if(d_in.begin(), d_in.end(), is_zero());

            // Get the new array size
            int new_size = new_end - d_in.begin();

            timer().endGpuTimer();

            thrust::copy(d_in.begin(), new_end, odata);

            return new_size;
        }
    }
}
