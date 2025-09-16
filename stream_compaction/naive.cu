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

        __global__ void kernNaiveScan(int n, int* odata, const int* idata, int stride)
        {
            int index = threadIdx.x + blockDim.x * blockIdx.x;
            if (index < n)
            {

                if (index >= (1 << stride))
                {
                    odata[index] = idata[index] + idata[index - (1 << stride)];
                }
                else
                {
                    odata[index] = idata[index];
                }
            }
        }

        __global__ void kernInclusiveToExclusive(int n, int* odata, const int* idata)
        {
            int index = threadIdx.x + blockDim.x * blockIdx.x;
            if (index == 0)
            {
                odata[0] = 0;
            }
            else if (index < n)
            {
                odata[index] = idata[index - 1];
            }
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
            // TODO
            //Starting Input
            int *dev_arrA;
            //Starting Output
            int *dev_arrB;

            cudaMalloc((void**)&dev_arrA, sizeof(int) * n);
            cudaMalloc((void**)&dev_arrB, sizeof(int) * n);

            cudaMemcpy(dev_arrA, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            

            int threadsPerBlock = 256;
            dim3 totalBlocks ((n + threadsPerBlock - 1) / threadsPerBlock);

            int log2Ceil = ilog2ceil(n);

            timer().startGpuTimer();
            for (int i = 0; i < log2Ceil; i++)
            {
                kernNaiveScan << <totalBlocks, threadsPerBlock >> > (n, dev_arrB, dev_arrA, i);
                std::swap(dev_arrA, dev_arrB);
            }
            
            kernInclusiveToExclusive <<<totalBlocks, threadsPerBlock >>> (n, dev_arrB, dev_arrA);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_arrB, sizeof(int) * n, cudaMemcpyDeviceToHost);
            


            cudaFree(dev_arrA);
            cudaFree(dev_arrB);

            
        }
    }
}
