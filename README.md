CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Zhuoran Li
  * [LinkedIn](https://www.linkedin.com/in/zhuoran-li-856658244/)
* Tested on: Windows 11, AMD Ryzen 5 5600H @ 3.30 GHz 16.0GB, NVIDIA GeForce RTX 3050 Laptop GPU 4GB

## 1. Overview
This project implements the parallel **prefix sum (scan)** algorithm (naive and work-efficient), as well as two of its applications: **stream compaction** and **radix sort**.

### cpu.cu
- `CPU::scan`, `CPU::compactWithoutScan` and `CPU::compactWithScan` provide serial implementations used as references for testing.

### naive.cu
- `Naive::scan`: Performs an exclusive scan within each block by iteratively doubling the offset and using shared memory as a ping-pong buffer. Each thread contributes by summing its own element with a neighbor at the current offset.

  **For arrays larger than a block**, I compute a block sum (the last element of a block's scan + the last element of the input array within the block) for each block and then scan the block sum array on the host. The resulting offsets are written back to the device and added to each block's output.
  

### efficient.cu
- `Efficient::scan`: Contains a binary tree-based algorithm, including an up-sweep and a down-sweep phase. Like the naive scan, it relies on shared memory for cross-block communication.

  Since the algorithm assumes a binary tree structure, we need to pad the array to the next power of 2. To do that, I allocated another array on device with padded size, initialize all elements with 0, and copy the initial input array into the padded array.
- `Efficient::compact`: The stream compaction algorithm first maps the input array into a boolean array, marking elements with 1 and 0. An exclusive scan is then performed on this boolean array to produce target indices for the valid elements. Finally, a scatter kernel places the valid elements into their new compacted positions, and the total count is obtained from the last scanned index plus the last boolean value.


## 2. Extra Features
### 2.1. Shared Memory Allocation
Both `Naive::scan` and `Efficient::scan` load the input array into shared memory to improve performance. Shared memory is allocated dynamically at kernel launch, with the required size specified as part of the kernel configuration.

Using `Efficient::scan()` as an example:

- **Single-block case:** If only *one* block is used, we allocate `2 * n * sizeof(int)` bytes of shared memory, where `n` is the array length.

- **General case:** For arrays of arbitrary length processed across multiple blocks, we instead allocate `2 * blockDim.x * sizeof(int)` bytes per block, which is how I implemented here.

```
kernEfficientScan<<<blocks, threads, 2 * threads * sizeof(int)>>>(pad_n, d_odata, d_idata, d_blockSum);
```
Inside the kernel, the input array is first copied into a shared memory buffer:
```
extern __shared__ int temp[];
temp[2 * thid] = idata[2 * id];
temp[2 * thid + 1] = idata[2 * id + 1];
```

Notice here `thid = threadIdx.x`, `id = threadIdx.x + blockIdx.x * blockDim.x`.

In the initial version, each thread simply loaded two consecutive elements into `temp`. In the optimized version, additional steps are taken to avoid bank conflicts (see [2.2. Avoid Bank Conflicts](#2.2.-avoid-bank-conflicts)).


### 2.2. Avoid Bank Conflicts
To avoid bank conflicts in `Efficient::scan`, we add a small offset to each index so that consecutive threads are distributed across different banks. Compared to the implementation in [GPU Gem 3 Ch 39-3](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda), the key difference lies in how the input array is loaded **across blocks**. 

The reference code uses an offset of `n / 2` when assigning indices. In contrast, my implementation uses `blockDim.x` as the offset. This adjustment is necessary because each thread must access elements based on its **global index** rather than just its local position within a block.

Accordingly, calculate indices relative to the start of each block:
```
int ai = thid;
int bi = thid + blockDim.x;
// ...
int blockStart = blockIdx.x * size;
temp[ai + bankOffsetA] = idata[blockStart + ai];
temp[bi + bankOffsetB] = idata[blockStart + bi];
```

### 2.3. Radix Sort
### 2.4. Thrust::remove_if


## 3. Performance Analysis