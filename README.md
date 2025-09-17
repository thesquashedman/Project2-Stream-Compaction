CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 2 - CUDA Stream Compaction**

* Pavel Peev
  * [LinkedIn](https://www.linkedin.com/in/pavel-peev-5568561b9/), [personal website](www.cartaphil.com)
* Tested on: Windows 11, i7-1270, NVIDIA T1000

### Description

Comparison of implementations of exclusive scans (and implementations of compact for the CPU and the work effient scan), comparing a basic CPU implementation, a naive CUDA implementation which adds an element stride spaces to the right over log2(n) kernal calls, the work efficient implmentation using an upsweep and downsweep stage to create a balanced binary tree, and thrust's own implementation of an exclusive scan.

### Sample Output
Below is an example of the output from running the test with array size 2^28
```
****************
** SCAN TESTS **
****************
    [  45  15  18  24  10  15  38  38  47  12   8  39  45 ...  11   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 421.69ms    (std::chrono Measured)
    [   0  45  60  78 102 112 127 165 203 250 262 270 309 ... -2015394307 -2015394296 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 419.546ms    (std::chrono Measured)
    [   0  45  60  78 102 112 127 165 203 250 262 270 309 ... -2015394410 -2015394369 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 520.167ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 519.373ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 392.432ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 392.508ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 16.1508ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 17.0412ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   3   2   1   3   0   2   1   0   0   0   3   3   2 ...   3   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 570.409ms    (std::chrono Measured)
    [   3   2   1   3   2   1   3   3   2   2   2   3   2 ...   3   3 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 576.547ms    (std::chrono Measured)
    [   3   2   1   3   2   1   3   3   2   2   2   3   2 ...   1   3 ]
    passed
==== cpu compact with scan ====
   elapsed time: 1481.78ms    (std::chrono Measured)
    [   3   2   1   3   2   1   3   3   2   2   2   3   2 ...   3   3 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 2605.5ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 2604.63ms    (CUDA Measured)
    passed
```
### Performance

<img width="1152" height="768" alt="Line Chart" src="https://github.com/user-attachments/assets/05998878-880b-41b3-86be-ba80e3f76e53" />



### Performance Analysis
Blocks of size 128 were used, as that seemed to perform the best across all the algorithms.

The efficient implementation performs marginally better than it's naive and cpu counterparts.The naive implementation performed the worst, which makes sense do to it's nlog2(n) complexity and the high cost for invoking kernal calls, making it slower than a regular CPU implemtation (and potentially the use of the older NVIDIA T1000 GPU). The thrust implementation performs significantly better than the others, showing just how much the exclusive scan algorithm can be optimized.


Analyzing the kernal invocations within NSight Compute (shown below), it becomes clear that the efficient implementation can be improved upon. Both the upsweep and downsweep have low memory throughput. Using shared memory, we can load the memory coherently for each of the blocks, which should reduce the amount of memory loads needed with the current algorithm.

<img width="1530" height="264" alt="image" src="https://github.com/user-attachments/assets/b4360172-0e8a-4a1a-b1df-0f3cabface03" />

I also use a modulo operator within the kernal, which is a well known expensive operation which could be replaced. Also, it is very likely that the warps diverge, with some warps having only 1 thread doing any work. With some clever indexing, this could be optimized so that all the working threads fall into the same warps, allowing for the warps with non working threads to retire early.

