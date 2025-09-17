#include <cstdio>
#include "cpu.h"
#include <iostream>

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int* odata, const int* idata) {
            timer().startCpuTimer();
            // TODO

            // -------------- Option 1: Single for loop ----------------
            odata[0] = 0;
            for (int i = 1; i < n; ++i) {
                odata[i] = odata[i - 1] + idata[i - 1];
            }

            // --------------- Option 2: Simulate GPU -------------------
            //int layer = ilog2ceil(n);

            //// Copy the input to the ping pong buffer
            //int* ppdata = new int[n];
            //for (int i = 0; i < n; ++i) {
            //    ppdata[i] = idata[i];
            //}

            //// Scan
            //for (int d = 1; d <= layer; ++d) {
            //    int offset = 1 << (d - 1);
            //    for (int i = 0; i < n; ++i) {
            //        if (i >= offset) {
            //            odata[i] = ppdata[i - offset] + ppdata[i];
            //        } else {
            //            odata[i] = ppdata[i];
            //        }
            //    }
            //    std::swap(odata, ppdata);
            //}
            //// Shift right
            //odata[0] = 0;
            //for (int i = 1; i < n; ++i) {
            //    odata[i] = ppdata[i - 1];
            //}
            //delete[] ppdata;

            // -----------------------------------------------------------
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int idx = 0;
            for (int i = 0; i < n; ++i) {
                if (idata[i] != 0) {
                    odata[idx] = idata[i];
                    idx++;
                }
            }

            timer().endCpuTimer();
            return idx;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            // Create a boolean array
            int* boolVal = new int[n];
            int cnt = 0;
            for (int i = 0; i < n; ++i) {
                if (idata[i] == 0) {
                    boolVal[i] = 0;
                } else {
                    boolVal[i] = 1;
                    cnt++;
                }
            }

            // Scan
            int* scanVal = new int[n];
            scanVal[0] = 0;
            for (int i = 1; i < n; ++i) {
                scanVal[i] = scanVal[i - 1] + boolVal[i - 1];
            }

            // Scatter
            for (int i = 0; i < n; ++i) {
                if (boolVal[i]) {
                    int idx = scanVal[i];
                    odata[idx] = idata[i];
                }
            }

            delete[] boolVal;
            delete[] scanVal;

            timer().endCpuTimer();
            return cnt;
        }
    }
}
