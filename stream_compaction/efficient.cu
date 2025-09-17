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
        
        __global__ void kernUpSweep(int n, int* data, int d)
        {
            int index = threadIdx.x + blockDim.x * blockIdx.x;
            if (index < n)
            {
                //Back when d was 0, 1, 2, 3...
                /*
                if (index % (int)powf(2, d + 1) == 0)
                {
                    data[index + (int)powf(2, d + 1) - 1] += data[index + (int)powf(2, d) - 1];

                }
                */
                if (index % (d * 2) == 0)
                {
                    data[index + d * 2 - 1] += data[index + d - 1];

                }
                
            }
        }
        __global__ void kernDownSweep(int n, int* data, int d)
        {
            int index = threadIdx.x + blockDim.x * blockIdx.x;
            if (index < n)
            {
                //Back when d was 0, 1, 2, 3...
                /*
                if (index  % (int)powf(2, d + 1) == 0)
                {
                    int t = data[index + (int)powf(2, d) - 1];
                    data[index + (int)powf(2, d) - 1] = data[index + (int)powf(2, d + 1) - 1];
                    data[index + (int)powf(2, d + 1) - 1] += t;
                }
                */
                if (index % (d * 2) == 0)
                {
                    int t = data[index + d - 1];
                    data[index + d - 1] = data[index + d * 2 - 1];
                    data[index + d * 2 - 1] += t;
                }
            }

        }

        __global__ void kernMapToBoolean(int n, int* oData, const int* iData)
        {
            int index = threadIdx.x + blockDim.x * blockIdx.x;
            if (index < n)
            {
                int val = iData[index];
                if (val != 0)
                {
                    oData[index] = 1;
                }
            }
        }
        __global__ void kernScatter(int n, int* oData, const int* iData, const int* boolArray, const int* scannedArray)
        {
            int index = threadIdx.x + blockDim.x * blockIdx.x;
            if (index < n)
            {
                if (boolArray[index])
                {
                    oData[scannedArray[index]] = iData[index];
                }
            }
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata, bool forCompact) {
           
            // TODO
            
            int* dev_arr;

            int log2Ceil = ilog2ceil(n);
            int arraySize = 1 << log2Ceil;

            cudaMalloc((void**)&dev_arr, sizeof(int) * arraySize);
            checkCUDAError("Bad Malloc");
           
            //Make sure that the array is filled with 0s to start with
            cudaMemset(dev_arr, 0, sizeof(int) * arraySize);
            checkCUDAError("Bad memset");
            if (!forCompact)
            {
                cudaMemcpy(dev_arr, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            }
            else
            {
                cudaMemcpy(dev_arr, idata, sizeof(int) * n, cudaMemcpyDeviceToDevice);
            }
            checkCUDAError("Bad copy of initial data");
            
            int threadsPerBlock = 128;
            dim3 totalBlocks((arraySize + threadsPerBlock - 1) / threadsPerBlock);
            if (!forCompact)
            {
                timer().startGpuTimer();
            }
            for (int d = 1; d < arraySize; d *= 2)
            {
                kernUpSweep << <totalBlocks, threadsPerBlock >> > (arraySize, dev_arr, d);
                checkCUDAError("up sweep failure");
            }
            cudaMemset(dev_arr + (arraySize - 1), 0, sizeof(int));
            checkCUDAError("Bad zeroing of last element");

            for (int d = arraySize / 2; d > 0; d /= 2)
            {
                kernDownSweep << <totalBlocks, threadsPerBlock >> > (arraySize, dev_arr, d);
                checkCUDAError("down sweep failure");
            }

            cudaDeviceSynchronize();
            if (!forCompact)
            {
                timer().endGpuTimer();
                cudaMemcpy(odata, dev_arr, sizeof(int) * n, cudaMemcpyDeviceToHost);
                
            }
            else
            {
                cudaMemcpy(odata, dev_arr, sizeof(int) * n, cudaMemcpyDeviceToDevice);
            }
            checkCUDAError("memcpy output failure");
            
            
            cudaFree(dev_arr);
            cudaDeviceSynchronize();
            
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

            
            int* dev_in;
            int* dev_out;
            int* dev_boolArray;
            int* dev_scannedArray;

            cudaMalloc((void**)&dev_in, sizeof(int) * n);
            cudaMalloc((void**)&dev_boolArray, sizeof(int) * n);
            cudaMalloc((void**)&dev_scannedArray, sizeof(int) * n);
            cudaMalloc((void**)&dev_out, sizeof(int) * n);
            checkCUDAError("Bad Malloc");


            cudaMemset(dev_boolArray, 0, sizeof(int) * n);
            checkCUDAError("Bad Memset");
            cudaMemcpy(dev_in, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("Bad Memcpy");

            int threadsPerBlock = 128;
            dim3 totalBlocks((n + threadsPerBlock - 1) / threadsPerBlock);

            timer().startGpuTimer();
            kernMapToBoolean << <totalBlocks, threadsPerBlock >> > (n, dev_boolArray, dev_in);


            scan(n, dev_scannedArray, dev_boolArray, true);

            

            int count;
            int lastElement;
            cudaMemcpy(&count, dev_scannedArray + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastElement, dev_boolArray + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            count += lastElement;

            kernScatter << <totalBlocks, threadsPerBlock >> > (n, dev_out, dev_in, dev_boolArray, dev_scannedArray);
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_out, sizeof(int) * n, cudaMemcpyDeviceToHost);

            cudaFree(dev_in);
            cudaFree(dev_out);
            cudaFree(dev_boolArray);
            cudaFree(dev_scannedArray);
            
            
            return count;
        }
    }
}
