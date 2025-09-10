#include <cstdio>
#include "cpu.h"

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
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            //Note, exclusive scan
            odata[0] = 0;
            for (int i = 1; i < n; i++)
            {
                odata[i] = odata[i - 1] + idata[i - 1];
            }
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

            //Points to the next spot for inserting into odata
            int oDataIndex = 0;

            for (int i = 0; i < n; i++)
            {
                if (idata[i] != 0)
                {
                    odata[oDataIndex] = idata[i];
                    oDataIndex++;
                }
            }


            timer().endCpuTimer();
            return oDataIndex;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO

            //Step 1: create bool array
            int* boolArray = new int[n] {-1};
            for (int i = 0; i < n; i++)
            {
                if (idata[i] == 0)
                {
                    boolArray[i] = 0;
                }
                else
                {
                    boolArray[i] = 1;
                }
            }
            //Step 2: Scan bool array
            //Note: not using scan since it's being timed with the same timer.
            int* scanArray = new int[n];
            scanArray[0] = 0;
            for (int i = 1; i < n; i++)
            {
                scanArray[i] = scanArray[i - 1] + boolArray[i - 1];
            }

            //Step 3: Scatter
            int count = 0;
            for (int i = 0; i < n; i++)
            {
                if (boolArray[i] == 1)
                {
                    count++;
                    odata[scanArray[i]] = idata[i];
                }
            }


           
            delete[] boolArray;
            delete[] scanArray;
            timer().endCpuTimer();
            return count;
        }
    }
}
