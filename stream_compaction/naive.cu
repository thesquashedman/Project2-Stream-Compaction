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
        //Starting Input
        int* dev_arrA;
        //Starting Output
        int* dev_arrB;


        // TODO: __global__

        __global__ void naiveScan(int n, int* odata, const int* idata, int stride)
        {
            int index = threadIdx.x + blockDim.x * blockIdx.x;
            if (index >= pow(2, stride - 1))
            {
                odata[index] = idata[index] + idata[index - (int)pow(2, stride - 1)];
            }
            else
            {
                odata[index] = idata[index];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            cudaMalloc((void**)dev_arrA, sizeof(int) * n);
            cudaMalloc((void**)dev_arrB, sizeof(int) * n);

            cudaMemcpy(dev_arrA, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            int threadsPerBlock = 128;
            dim3 totalBlocks ((n + threadsPerBlock - 1) / threadsPerBlock);

            int log2Ceil = ilog2ceil(n);
            for (int i = 0; i < log2Ceil; i++)
            {
                naiveScan << <totalBlocks, threadsPerBlock >> > (n, dev_arrA, dev_arrB, i);
                std::swap(dev_arrA, dev_arrB);
            }
            
            if (log2Ceil % 2 == 1)
            {
                std::swap(dev_arrA, dev_arrB);
            }

            cudaMemcpy(odata, dev_arrB, sizeof(int) * n, cudaMemcpyDeviceToHost);

            cudaFree(dev_arrA);
            cudaFree(dev_arrB);

            timer().endGpuTimer();
        }
    }
}
