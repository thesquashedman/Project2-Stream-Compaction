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
        
        __device__ void upSweep(int index, int* data, int log2Ceil)
        {
            
            for (int d = 0; d < log2Ceil; d++)
            {
                if ((index + 1) % (int)powf(2, d + 1))
                {
                    data[index] += data[index - (int)powf(2, d)];
                }
                __syncthreads();
            }
            
        }
        __device__ void downSweep(int index, int* data, int log2Ceil)
        {
            
            
            for (int d = log2Ceil; d >= 0; d--)
            {
                if ((index + 1) % (int)powf(2, d + 1))
                {
                    int t = data[index + (int)powf(2, d) - 1];
                    data[index + (int)powf(2, d) - 1] = data[index + (int)powf(2, d + 1)];
                    data[index + (int)powf(2, d + 1)] += t;
                }
                __syncthreads();
            }

        }

        __global__ void kernEfficientScan(int n, int* data, int log2Ceil)
        {
            int index = threadIdx.x + blockDim.x * blockIdx.x;
            if (index < n)
            {
                
                upSweep(index, data, log2Ceil);
                if (index == n - 1)
                {
                    data[index] = 0;
                }
                downSweep(index, data, log2Ceil);
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
           
            // TODO
            int* dev_arr;

            int log2Ceil = ilog2ceil(n);
            int arraySize = powf(2, log2Ceil);

            cudaMalloc((void**)&dev_arr, sizeof(int) * arraySize);
            //Make sure that the array is filled with 0s to start with
            cudaMemset(dev_arr, 0, sizeof(int) * arraySize);
            cudaMemcpy(dev_arr, idata, sizeof(int) * arraySize, cudaMemcpyHostToDevice);

            int threadsPerBlock = 128;
            dim3 totalBlocks((arraySize + threadsPerBlock - 1) / threadsPerBlock);

            timer().startGpuTimer();

            kernEfficientScan << <totalBlocks, threadsPerBlock >> > (arraySize, dev_arr, log2Ceil);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_arr, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(dev_arr);

            
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
