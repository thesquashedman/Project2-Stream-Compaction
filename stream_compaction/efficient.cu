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
        
        __device__ void UpSweep(int index, int* data, int iLogCeil)
        {
            
            for (int d = 0; d < iLogCeil; d++)
            {
                if ((index + 1) % (int)powf(2, d + 1))
                {
                    data[index] += data[index - (int)powf(2, d)];
                }
                cudaDeviceSynchronize();
            }
            
        }
        __device__ void DownSweep(int n, int index, int* data, int iLogCeil)
        {
            if (index == n - 1)
            {
                data[index] = 0;
            }
            
            for (int d = iLogCeil; d >= 0; d--)
            {
                if ((index + 1) % (int)powf(2, d + 1))
                {
                    int t = data[index + (int)powf(2, d) - 1];
                    data[index + (int)powf(2, d) - 1] = data[index + (int)powf(2, d + 1)];
                    data[index + (int)powf(2, d + 1)] += t;
                }
                cudaDeviceSynchronize();
            }

        }

        __global__ void kernEfficientSwap(int n, int* data)
        {

        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            int* dev_arr;

            int log2Ceil = ilog2ceil(n);
            int arraySize = powf(2, n);

            cudaMalloc((void**)&dev_arr, sizeof(int) * n);
            timer().endGpuTimer();
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
            timer().endGpuTimer();
            return -1;
        }
    }
}
